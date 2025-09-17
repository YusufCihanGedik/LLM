import os 
import dotenv
dotenv.load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

# Adım 1: LLM Kurulumu
llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

template = "Kısa ve anlaşılır {user_message} açıkla"

prompt = PromptTemplate(
    input_variables=["user_message"], #Kullanıcıdan alınan veya koda tanımlanan konu
    template=template
)

konu = "Yapay Zeka Nedir"
prompt_message = prompt.format(user_message=konu)

#llm invoke verilen mesajları llm gönderir modelden yanıt alır
response = llm.invoke([HumanMessage(content=prompt_message)])

print(response.content)


"""
[1] Kullanıcı girişi
"Yapay Zeka Nedir"
        │
        ▼
[2] PromptTemplate hazırlanır
"Kısa ve anlaşılır {user_message} açıkla"
        │
        ▼
[3] format() çağrısı ile doldurulur
"Kısa ve anlaşılır Yapay Zeka Nedir açıkla"
        │
        ▼
[4] HumanMessage oluşturulur
HumanMessage(content="Kısa ve anlaşılır Yapay Zeka Nedir açıkla")
        │
        ▼
[5] LLM'e gönderilir (llm.invoke)
        │
        ▼
[6] Model cevap üretir
"Yapay zeka, bilgisayarların insan benzeri öğrenme, düşünme ve problem çözme yeteneklerini taklit etmesidir."
        │
        ▼
[7] Çıktı ekrana yazılır

"""

