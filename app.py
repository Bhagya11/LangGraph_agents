# app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama

# -----------------------------
# Load API Keys
# -----------------------------
load_dotenv('.env')

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if not WEATHER_API_KEY or not TAVILY_API_KEY:
    st.error("Missing WEATHER_API_KEY or TAVILY_API_KEY in your .env file.")
    st.stop()

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

# -----------------------------
# Tool Functions
# -----------------------------
def get_weather(query: str):
    """Fetch current weather data for a location."""
    endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
    response = requests.get(endpoint)
    return response.json()

def search_web(query: str):
    """Search the web using Tavily."""
    tavily_search = TavilySearchResults(
        api_key=TAVILY_API_KEY, 
        max_results=2, 
        search_depth='advanced', 
        max_tokens=1000
    )
    return tavily_search.invoke(query)

def agent(query: str):
    """Route the query to weather, web search, or LLM."""
    query_lower = query.lower()
    if "weather" in query_lower or "rain" in query_lower or "temperature" in query_lower:
        return ("weather", get_weather(query))
    elif "search" in query_lower or "tell me about" in query_lower:
        return ("search", search_web(query))
    else:
        response = llm.invoke(query)
        return ("chat", response.content)

# -----------------------------
# Streamlit Frontend
# -----------------------------
st.set_page_config(page_title="AI Agent", page_icon="🤖")

st.title("🤖 AI Agent: Weather, Search & Chat")
st.write("Ask me about weather, search topics, or chat!")

# Chat input
query = st.text_input("Enter your query:")

if query:
    with st.spinner("Thinking..."):
        mode, result = agent(query)

    st.subheader("Response:")

    if mode == "weather":
        # Pretty weather display
        if "current" in result:
            location = result["location"]["name"]
            region = result["location"]["region"]
            country = result["location"]["country"]
            temp_c = result["current"]["temp_c"]
            condition = result["current"]["condition"]["text"]
            icon = result["current"]["condition"]["icon"]

            st.markdown(f"### 🌍 {location}, {region}, {country}")
            st.image(f"http:{icon}", width=80)
            st.markdown(f"**Temperature:** {temp_c}°C")
            st.markdown(f"**Condition:** {condition}")
        else:
            st.error("Could not fetch weather details.")
    
    elif mode == "search":
        # Show search results nicely
        if isinstance(result, list):
            for idx, r in enumerate(result):
                st.markdown(f"**🔍 Result {idx+1}:** {r}")
        else:
            st.json(result)
    
    else:
        # Chat mode
        st.write(result)
