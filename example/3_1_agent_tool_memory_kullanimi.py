import os 
import dotenv
dotenv.load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


llm = ChatOpenAI()

def calculator_tool(x: str) -> str:
    return str(eval(x))

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)


tools =[
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Matematiksel işlemler yapmak için kullanılır. Örnek: 2+2, 3*5"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)



while True:
    user_input = input("Soru yazın (çıkmak için 'q'): ")
    if user_input.lower() in ["q", "quit", "exit"]:
        break
    yanit = agent.invoke({"input": user_input})
    print("Yanıt:", yanit["output"])