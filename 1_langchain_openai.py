import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Adım 1: LLM Kurulumu
llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

# Basit bir sohbet örneği
# HumanMessage, LangChain kütüphanesinde insan tarafından yazılmış bir mesajı temsil eden bir sınıftır.
# user_message = HumanMessage(content="Hello, how are you?")

while True:
    user_input = input("Bir soru yazın: ")
    user_message = HumanMessage(content=user_input)

    # Adım 2: LLM'yi çağırma
    response = llm.invoke([user_message])

    # Adım 3: Yanıtı yazdırma
    print(response.content)