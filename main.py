import os
from typing import List, Union
import streamlit as st
import graphviz
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
# from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage

import pandas as pd
# load_dotenv()
from cypher_chain import CYPHER_QA_PROMPT
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Stock Data Chatbot")

from cypher_chain import CustomCypherChain

# Load secrets
url = "neo4j+s://c1e96139.databases.neo4j.io"
username = "neo4j"
password = "t90Pab_G9IVD3nQ7QWN_gX5l9IQZ0q9kM4fbk5iCX7A"

# Langchain x Neo4j connections
graph = Neo4jGraph(username=username, password=password, url=url)

graph_search = None

# Session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = []

if "viz_data" not in st.session_state:
    st.session_state["viz_data"] = []

if "database" not in st.session_state:
    st.session_state["database"] = []

if "cypher" not in st.session_state:
    st.session_state["cypher"] = []


def generate_context(prompt: str, context_data: str = "generated") -> List[Union[AIMessage, HumanMessage]]:
    context = []
    # Add all history to context
    size = len(st.session_state["generated"])
    for i in range(size):
        context.append(HumanMessage(content=st.session_state["user_input"][i]))
        context.append(AIMessage(content=st.session_state[context_data][i]))
    # Add the latest user prompt
    context.append(HumanMessage(content=str(prompt)))
    return context


def dynamic_response_tabs(i):
    tabs_to_add = ["💬Chat"]
    data_check = {
        "🔍Cypher": bool(st.session_state["cypher"][i]),
        "🗃️Database results": isinstance(st.session_state["database"][i], pd.DataFrame) and not st.session_state["database"][i].empty,
        "🕸️Visualization": st.session_state["viz_data"][i] is not None
    }

    for tab_name, has_data in data_check.items():
        if has_data:
            tabs_to_add.append(tab_name)

    with st.chat_message("user"):
        st.write(st.session_state["user_input"][i])

    with st.chat_message("assistant"):
        selected_tabs = st.tabs(tabs_to_add)

        with selected_tabs[0]:
            st.write(st.session_state["generated"][i])
        if len(selected_tabs) > 1:
            with selected_tabs[1]:
                st.code(st.session_state["cypher"][i], language="cypher")
        if len(selected_tabs) > 2:
            with selected_tabs[2]:
                # Display database results in a table
                database_results = st.session_state["database"][i]
                if isinstance(database_results, pd.DataFrame) and not database_results.empty:
                    st.table(database_results)  # Use st.table() for static table
                    # st.dataframe(database_results)  # Use st.dataframe() for interactive table
                else:
                    st.write("No data found.")
        if len(selected_tabs) > 3:
            with selected_tabs[3]:
                graph_object = graphviz.Digraph()
                for final_entity in st.session_state["viz_data"][i][1]:
                    graph_object.node(final_entity, fillcolor="lightblue", style="filled")
                for record in st.session_state["viz_data"][i][0]:
                    graph_object.edge(record["source"], record["target"], label=record["type"])
                st.graphviz_chart(graph_object)


def get_text() -> str:
    input_text = st.chat_input("Ask a question about stocks...")
    return input_text


openai_api_key = "sk-None-5gvwQgiGTcUl9O3uV9NoT3BlbkFJWAvwkGDT83BDZhLqy06s"
os.environ["OPENAI_API_KEY"] = openai_api_key
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
if openai_api_key:
    graph_search = CustomCypherChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0.0, model_name="gpt-4"),
        qa_llm=ChatOpenAI(temperature=0.0),
        graph=graph,
        qa_prompt=CYPHER_QA_PROMPT,
    )

csv_file_path = 'fewshot.csv'
df = pd.read_csv(csv_file_path)
questions = df['Question'].tolist()
st.sidebar.markdown("## Example questions")
for i, question in enumerate(questions, 1):
    st.sidebar.markdown(f"{i}. {question}")
    
user_input = get_text()

if user_input:
    with st.spinner("Processing"):
        context = generate_context(user_input)
        output = graph_search({"query": user_input, "chat_history": context})

        st.session_state.user_input.append(user_input)
        st.session_state.generated.append(output["result"])
        st.session_state.viz_data.append(output.get("viz_data", None))
        # Assuming the database output is a list of dicts, convert to DataFrame
        db_output = output.get("database", [])
        
        # Debug: Check the length of the database output
        # st.write(f"Number of records fetched: {len(db_output)}")
        
        if db_output:
            st.session_state.database.append(pd.DataFrame(db_output))
            st.table(pd.DataFrame(db_output))  # Use st.table() for static table
        else:
            st.session_state.database.append(pd.DataFrame())
        st.session_state.cypher.append(output.get("cypher", ""))

if st.session_state["generated"]:
    size = len(st.session_state["generated"])
    # Display all exchanges
    for i in range(size):
        dynamic_response_tabs(i)
