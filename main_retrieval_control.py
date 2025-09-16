import os 
from dotenv import load_dotenv
load_dotenv()
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

company_table = [
    {"id": 1, "name": "John Doe", "department": "Engineering", "role": "Software Engineer"},
    {"id": 2, "name": "Jane Smith", "department": "Marketing", "role": "Marketing Manager"},
    {"id": 3, "name": "Alice Johnson", "department": "HR", "role": "HR Specialist"},
    {"id": 4, "name": "Bob Wilson", "department": "Engineering", "role": "DevOps Engineer"},
]


#Verilerin Document nesnesine dönüştürülmesi
documents = []
for raw in company_table:
    content = f"ID: {raw['id']}, Name: {raw['name']}, Department: {raw['department']}, Role: {raw['role']}"
    metadata = {"id": raw['id'], "department": raw['department'], "role": raw['role']}
    #Document içerisine content ve metadata ekleniyor
    documents.append(Document(page_content=content, metadata=metadata))



#Embedding modeli
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embedding_model)
print("\nVektör veritabanı oluşturuldu ve metinler yüklendi.")

#LLM ve Bellek


llm = ChatOpenAI(
    model="gpt-5-mini")

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    return_messages=True
)

template = """Sen, şirket çalışanları hakkında bilgi veren bir İK asistanısın.
Sana verilen Bağlam'ı ve Sohbet Geçmişi'ni kullanarak kullanıcının sorusunu cevapla.
Eğer bilgi bağlamda yoksa, 'Bu bilgiye sahip değilim' de.

Sohbet Geçmişi:
{history}

Bağlam:
{context}

Kullanıcı: {input}
Asistan:"""

prompt = PromptTemplate(   
    input_variables=["history", "context", "input"],
    template=template
)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

print("\n--- Şirket İK Asistanı ---")
while True:
    query = input("Kullanıcı: ")
    if query.lower() in ["exit", "quit"]:
        print("Asistan: Görüşmek üzere!")
        break

    # Soruya uygun belgeleri vektör veritabanından al
    relevant_docs = vector_db.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Zinciri çalıştır ve cevap al
    response = conversation.invoke({"input": query, "context": context}) 
    print("Asistan:", response["text"]) 
    
""""
Veri: Yapılandırılmış bir çalışan listesi.
Doküman Dönüştürücü: Bu listeyi LangChain Document nesnelerine çeviren bir döngü.
Vektör Deposu: FAISS - Hızlı, bellek-içi anlamsal arama için.
LLM: Ollama - API maliyeti olmadan yerel olarak çalışan bir dil modeli.
Hafıza: ConversationBufferMemory - Sohbet geçmişini tutmak için.
Prompt Şablonu: context, history ve input değişkenlerini açıkça kabul eden özel bir şablon.
Zincir: LLMChain veya ConversationChain gibi temel bir zincir.

"""