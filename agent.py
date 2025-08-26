
#ENVIRONMENT SETUP
import os
from dotenv import load_dotenv

load_dotenv()

#LLM and TOOLS Setup
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

groq_LLM = ChatGroq(model = "openai/gpt-oss-20b", api_key = os.getenv("GROQ_API_KEY"))
Local_LLM = ChatOllama(model = "gemma3:1b")

Search_Tool = TavilySearchResults(max_results = 2)

#langgraph setup
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=groq_LLM
    elif provider=="Local Model":
        llm=Local_LLM

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]

