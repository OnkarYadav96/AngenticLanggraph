from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()
import asyncio

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"], ##Ensure correct absolute path
                 "transport":"stdio"
            }
        },
        {
            "weather":{
                "url":"http://localhost:8000/mcp",  #Ensure server is running
                "transport":"streamable_http",
            }
        }
    )
    
    
    
    
    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
    
    
    tools=await client.get_tools()
    model=ChatGroq(model="qwen/qwen3.6-27b")
    agent=create_agent(
        model,tools
    )
    
    math_response=await agent.ainvoke(
        {"messages":[{"role":"user","content":"what is (3+5)*12?"}]}
    )
    
    weather_response=await agent.ainvoke(
            {"messages":[{"role":"user","content":"what is weather in california?"}]}
    )
    
    print("Math response",math_response["messages"][-1].content)
    print("Math response",weather_response["messages"][-1].content)  
    
    
    asyncio.run(main())