import os
import requests
from typing import Any, Dict, List, Optional, Tuple
from langchain.chains.graph_qa.cypher import extract_cypher # type: ignore
from langchain.chains.openai_functions import create_structured_output_chain # type: ignore
from langchain.schema import SystemMessage # type: ignore
from langchain.prompts import ChatPromptTemplate, PromptTemplate # type: ignore
from typing import List, Tuple

from langchain.chains import GraphCypherQAChain
from langchain.callbacks.manager import CallbackManagerForChainRun

try:
    from pydantic.v1.main import BaseModel, Field
except ImportError:
    from pydantic.main import BaseModel, Field

from cypher_validator import CypherQueryCorrector, Schema


def remove_entities(doc):
    """
    Replace named entities in the given text with their corresponding entity labels.

    Parameters:
    - doc (Spacy Document): processed SpaCy document of the input text.

    Returns:
    - str: The modified text with named entities replaced by their entity labels.

    Example:
    >>> replace_entities_with_labels("Apple is looking at buying U.K. startup for $1 billion.")
    'ORG is looking at buying GPE startup for MONEY .'
    """
    # Initialize an empty list to store the new tokens
    new_tokens = []
    # Keep track of the end index of the last entity
    last_end = 0

    # Iterate through entities, replacing them with their entity label
    for ent in doc.ents:
        # Add the tokens that come before this entity
        new_tokens.extend([token.text for token in doc[last_end : ent.start]])
        # Replace the entity with its label
        new_tokens.append(f"{ent.label_}")
        # Update the last entity end index
        last_end = ent.end

    # Add any remaining tokens after the last entity
    new_tokens.extend([token.text for token in doc[last_end:]])
    # Join the new tokens into a single string
    new_text = " ".join(new_tokens)
    return new_text


AVAILABLE_RELATIONSHIPS = [
    "HAS_PERIOD"
] 


CYPHER_SYSTEM_TEMPLATE = """
Purpose:
Your role is to convert user questions concerning data in a Neo4j database into accurate Cypher queries and display the output that contains more than one data in tabular format.

"""

cypher_query_corrector = CypherQueryCorrector(AVAILABLE_RELATIONSHIPS)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
Give the results that contains more than one data in Tabular format by default.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Even if the question doesn't provide full person or organization names, you should use the full names from the provided
information to construct an answer. You should have the ability to analyse and predict the future values. If i tell total profit and loss for the month
of july you should go to the neo4j and find equivalent month number for example july-7, june-6 etc..
you are  smart enough to understand the queries for other months by yourself
all the cypher queries are in fewshot.csv file.. you should analyse the query and fetch data fron neo4j and provide output.. similarly for other months ypu should analyse by yourself and provide output
Information:
{context}

Question: {question}
Helpful Answer:"""
CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)


class Entities(BaseModel):
    """Identifying information about entities."""

    name: List[str] = Field(
        ...,
        description="All the orders, profits, or date that appear in the text",
    )


class CustomCypherChain(GraphCypherQAChain):
    def process_entities(self, text: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting stock data entities from the text",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following input: Date {Date}, PNL {PNL}, ID {ID}, Qty {Qty}, Buy Price {BuyPrice}, Sell Price {SellPrice}, PNL {PNL}, Stop Loss {StopLoss}, Target {Target}, Buy Time {BuyTime}, Sell Time {SellTime}, Scrip {Scrip}, Highest Price Reached {HighestPriceReached}, Target Revised Count {TargetRevisedCount}",
                ),
            ]
        )

        entity_chain = create_structured_output_chain(
            Entities, self.qa_chain.llm, prompt
        )
        entities = entity_chain.run(text)
        print(entities)
        return entities.name

    def get_viz_data(self, entities: List[str]) -> List[Tuple[str, str]]:
        viz_query = """
        MATCH (o:AlgoOrders)-[r:RELATED_TO]->(p:Period)
        WHERE p.month IN $entities OR p.year IN $entities
        RETURN o.id AS source, type(r) AS type, p.month + "/" + p.year AS target
        LIMIT 5
        UNION
        MATCH (o:AlgoOrders)<-[r:RELATED_TO]-(p:Period)
        WHERE p.month IN $entities OR p.year IN $entities
        RETURN p.month + "/" + p.year AS source, type(r) AS type, o.id AS target
        LIMIT 5
        """
        results = self.graph.query(viz_query, {"entities": entities})
        return results

    def find_entity_match(self, entity: str, k: int = 3) -> List[str]:
        fts_query = """
        CALL db.index.fulltext.queryNodes('entity', $entity + "*", {limit:$k})
        YIELD node, AlgoOrders
        RETURN node.name AS result
        """
        results = self.graph.query(fts_query, {"entity": entity, "k": k})
        return [record["result"] for record in results]

    def generate_system_message(
        self, relevant_entities: List[str] = None, fewshot_examples: List[str] = None
    ) -> SystemMessage:
        system_message = CYPHER_SYSTEM_TEMPLATE
        system_message += (
            f"Database Schema: Please refer to the provided database schema {self.graph_schema} for reference. "
            "Guidelines: Relationships & Properties: Utilize only the relationship types "
            "and properties specified in the provided schema. Do not introduce new ones.\n"
        )
        if relevant_entities:
            system_message += (
                f"Entity Substitution: If the question mentions specific stocks, replace them in the query with corresponding stocks from "
                f"the given list. Given list of stocks is: {relevant_entities}\n"
                "Example: If the list contains BANKNIFTY12JUN24P49800: ['BANKNIFTY12JUN24P49800', 'BANKNIFTY12JUN24C49700'], replace 'BANKNIFTY12JUN24P49800' in the query with 'BANKNIFTY12JUN24P49800' or 'BANKNIFTY12JUN24C49700'.\n"
                "Flexible Queries: Ensure your Cypher queries can capture all relevant stocks.\n"
                "Correct: MATCH (t:Trade) WHERE t.scrip IN ['BANKNIFTY12JUN24P49800', 'BANKNIFTY12JUN24C49700'] MATCH (t)-[:BOUGHT]->(o:Option)"
            )
        if fewshot_examples:
            system_message += (
                f"Example Queries: Please refer to the provided example queries for constructing Cypher statements:\n"
                f"{fewshot_examples}\n"
            )
        relevant_entities = ["BANKNIFTY12JUN24P49800", "BANKNIFTY12JUN24C49700"]
        fewshot_examples = """
       MATCH (order:AlgoOrders)
	WITH order, toInteger(substring(order.order_buy_datetime, 6, 2)) AS orderMonth
	WHERE orderMonth = 7
	RETURN order
	MATCH (order:AlgoOrders)
	WITH order, toInteger(substring(order.order_buy_datetime, 6, 2)) AS orderMonth
	WHERE orderMonth = 7
	RETURN sum(order.profit_loss) AS totalProfitLoss
	WITH {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'} AS months
	MATCH (order:AlgoOrders)
	WITH order, toInteger(substring(order.order_buy_datetime, 6, 2)) AS orderMonth, toInteger(substring(order.order_buy_datetime, 9, 2)) AS orderDay, months
	WHERE months[orderMonth] = 'July' AND orderDay >= 1 AND orderDay <= 7
	RETURN order
        """
        
        return system_message
  

def get_fewshot_examples(self, question):
    # Make a request to the OpenAI API to generate embedding for the question
    response = requests.post(
        "https://api.openai.com/v1/engines/davinci-codex/completions",
        headers={
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
            "Content-Type": "application/json"
        },
        json={
            "prompt": question,
            "max_tokens": 50,  # Adjust as needed
            "model": "davinci-codex"
        }
    )
    
    # Extract the embedding from the response
    embedding = response.json()["choices"][0]["metadata"]["embedding"] # Assuming embedding is under metadata key
    
    # Use the embedding to query nodes in Neo4j
    results = self.graph.query(
        """
        CALL db.index.vector.queryNodes('fewshot', 3, $embedding)
        YIELD node, score
        RETURN node.Question AS question, node.Cypher as cypher
        """,
        {"embedding": embedding},
    )
    
    return results

    fewshot = "\n".join([f"#{el['question']}\n{el['cypher']}" for el in results])
    print("-" * 30)
    print(fewshot)
    return fewshot

def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> Dict[str, Any]:

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        chat_history = inputs["chat_history"]
        # Extract mentioned people and organizations and match them to database values
        entities = self.process_entities(question)
        print(f"NER found: {entities}")
        relevant_entities = dict()
        for entity in entities:
            relevant_entities[entity] = self.find_entity_match(entity)
        print(f"Relevant entities are: {relevant_entities}")

        # Get few-shot examples using vector search
        fewshots = self.get_fewshot_examples(question)

        system = self.generate_system_message(str(relevant_entities), fewshots)
        generated_cypher = self.cypher_generation_chain.llm.predict_messages(
            [system] + chat_history
        )
        print(generated_cypher.content)
        generated_cypher = extract_cypher(generated_cypher.content)
        validated_cypher = cypher_query_corrector(
            generated_cypher
        )
        print(validated_cypher)
        # If Cypher statement wasn't generated
        # Usually happens when LLM decides it can't answer
        if not "RETURN" in validated_cypher:
            chain_result: Dict[str, Any] = {
                self.output_key: validated_cypher,
                "viz_data": (None, None),
                "database": None,
                "cypher": None,
            }
            return chain_result

        # Retrieve and limit the number of results
        context = self.graph.query(
            validated_cypher, {"openai_api_key": os.environ["OPENAI_API_KEY"]}
        )[: self.top_k]

        result = self.qa_chain(
            {"question": question, "context": context}, callbacks=callbacks
        )
        final_result = result[self.qa_chain.output_key]

        final_entities = self.process_entities(final_result)
        if final_entities:
            viz_data = self.get_viz_data(final_entities)
        else:
            viz_data = None

        chain_result: Dict[str, Any] = {
            self.output_key: final_result,
            "viz_data": (viz_data, final_entities),
            "database": context,
            "cypher": validated_cypher,
        }
        return chain_result
