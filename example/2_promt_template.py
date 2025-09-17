"""
LLM’e gönderilecek promptları şablon şeklinde hazırlamak için kullanılan sınıftır.
"""
from langchain.prompts import PromptTemplate
#Örnek Kullanım
# template = "Your name is {name}. Age is {age}."
# prompt = PromptTemplate(
#     input_variables=["name", "age"],
#     template=template
# )

# print(prompt.format(name="Ahmet", age=30))

#Komplike Örnek

template = """
Aşşağıda kullanıcının sorusu var :
{question}

Sohbet geçmişi:
{chat_history}

verilen Bağlam:
{context}

"""
prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template=template   
)

print(prompt.format(
    question="2023 izin politikası nedir?",
    chat_history="Kullanıcı daha önce maaşları sormuştu.",
    context="2023 yılında çalışanlara 14 gün izin verilmiştir."
))