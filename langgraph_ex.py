# langgraph_ex.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# -----------------------------
# Load API Keys
# -----------------------------
load_dotenv(".env")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOllama(model="gemma3:1b", temperature=0)

# -----------------------------
# Tools
# -----------------------------
def get_weather(query: str):
    endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
    response = requests.get(endpoint)
    return response.json()

def search_web(query: str):
    tavily = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2)
    return tavily.invoke(query)

# -----------------------------
# Graph State
# -----------------------------
class AgentState(dict):
    query: str
    intent: str
    result: str

# -----------------------------
# Graph Nodes
# -----------------------------
def classify(state: AgentState):
    """Classify user query into intent."""
    query = state["query"].lower()
    if any(word in query for word in ["weather", "rain", "temperature"]):
        intent = "weather"
    elif any(word in query for word in ["search", "tell me about"]):
        intent = "search"
    else:
        intent = "chat"
    return {"intent": intent}

def handle_weather(state: AgentState):
    result = get_weather(state["query"])
    return {"result": result}

def handle_search(state: AgentState):
    result = search_web(state["query"])
    return {"result": result}

def handle_chat(state: AgentState):
    response = llm.invoke(state["query"])
    return {"result": response.content}

# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("classify", classify)
graph.add_node("weather", handle_weather)
graph.add_node("search", handle_search)
graph.add_node("chat", handle_chat)

graph.add_edge(START, "classify")

# Conditional branching
graph.add_conditional_edges(
    "classify",
    lambda state: state["intent"],
    {
        "weather": "weather",
        "search": "search",
        "chat": "chat"
    }
)

graph.add_edge("weather", END)
graph.add_edge("search", END)
graph.add_edge("chat", END)

app = graph.compile()

# -----------------------------
# Streamlit Frontend
# -----------------------------
st.set_page_config(page_title="LangGraph AI Agent", page_icon="🤖")
st.title("🤖 LangGraph AI Agent")

query = st.text_input("Enter your query:")

if query:
    with st.spinner("Thinking..."):
        result = app.invoke({"query": query})
    st.subheader("Response:")
    st.write(result["result"])
