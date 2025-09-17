"""
Amacımız langchain ile tool kullanımı
bunu yaparken llm destekli bir ajanın doğru aracı kullanılmasını sağlamamız gerekiyor
"zero-shot" agent kullanarak  yeni aracı tanımlardan agentin doğru aracı seçmesini sağlayacağız
"""
import os
import dotenv
dotenv.load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool ,AgentExecutor
from langchain.agents import AgentType
from langchain.tools import tool

# Adım 1: LLM Kurulumu
llm = ChatOpenAI()

# Adım 2: Araçların Tanımlanması (tool tanımı)
@tool
def calculator_tool(x: str) -> str:
    """Matematiksel işlemler yapmak için kullanılır. Örnek: 2+2, 3*5"""
    return str(eval(x))

#Birden fazla tool kullanacaksak liste içine ekliyoruz
tools = [calculator_tool]
# tools =[
#     Tool(
#         name="Calculator",
#         func=calculator_tool,
#         description="Matematiksel işlemler yapmak için kullanılır. Örnek: 2+2, 3*5"
#     )   
# ]

# Adım 3: Ajanın Oluşturulması
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # agent_type="zero-shot-react-description", # AgentType.ZERO_SHOT_REACT_DESCRIPTION aynı şeyi ifade eder
    verbose=True
)   

print(agent.invoke("3 ile 5'in çarpımı nedir?"))