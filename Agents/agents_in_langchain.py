from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain import hub
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import sys
import requests

# Load variables from .env if present
load_dotenv()

# Get Groq API token from env variable
groq_api_key = os.getenv("GROQ_API_KEY")
weatherstack_api_key = os.getenv("WEATHERSTACK_API_KEY")

if not groq_api_key:
    print("ERROR: Groq API key not found.")
    sys.exit(1)
else:
    print(" Groq API key loaded successfully.")

if not weatherstack_api_key:
    print("ERROR: WeatherStack API key not found.")
    sys.exit(1)
else:
    print(" WeatherStack API key loaded successfully.")

#  Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",  
    temperature=0
)

# create a search tool
search_tool = DuckDuckGoSearchRun()

# Weather tool
@tool
def get_weather(city: str) -> str:
    """Fetches the current weather data for a given city."""
    url = f'https://api.weatherstack.com/current?access_key={weatherstack_api_key}&query={city}'
    response = requests.get(url).json()
    return response

# Pull the ReAct prompt from langchain hub
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(llm=llm, tools=[search_tool, get_weather], prompt=prompt)

# Wrap it in AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=[search_tool, get_weather], verbose=True, handle_parsing_errors=True)

# Invoke the agent
response = agent_executor.invoke({"input": "What is the Sub Capital of Jharkhand and what is the weather there?"})

# Print the response
print(response)
