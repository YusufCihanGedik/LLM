import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory # <-- YENİ BİLEŞEN:
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter # Daha akıllı splitter'a geçiyoruz
from langchain_community.document_loaders import PyPDFLoader # <-- YENİ BİLEŞEN: PDF Yükleyici
from langchain.retrievers.self_query.base import SelfQueryRetriever # <-- YENİ RETRIEVER
from langchain.chains.query_constructor.base import AttributeInfo # <-- Metadata tanımlamak için
from langchain.schema import Document #Document nesnelerini manuel oluşturmak için 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

docs_with_metadata = [
    Document(
        page_content="2023 İK Politikası: Tüm çalışanların yıllık 14 gün izni vardır. 5 yılı dolduranlar 20 gün alır.",
        metadata={"year": 2023, "topic": "İK"}
    ),
    Document(
        page_content="2024 İK Politikası: Yeni politikaya göre, tüm çalışanların yıllık 16 gün izni vardır. 5 yılı dolduranlar ise 22 gün alır. Uzaktan çalışma haftada 3 güne çıkarılmıştır.",
        metadata={"year": 2024, "topic": "İK"}
    ),
    Document(
        page_content="2023 Finans Raporu: Şirket karı %10 artmıştır.",
        metadata={"year": 2023, "topic": "Finans"}
    ),
    Document(
        page_content="2024 Finans Raporu: Yeni yatırımlar sayesinde şirket karı %15 artmıştır.",
        metadata={"year": 2024, "topic": "Finans"}
    ),
    Document(
        page_content="Jon Doe eşi Jane Doe ile birlikte 2024 yılında şirkete katıldı.",
        metadata={"year": 2024, "topic": "Finans"}
    ),
    Document(
        page_content="Jane Doe, 2024 yılında şirkete katıldı.",
        metadata={"year": 2024, "topic": "Finans"}
    )
]

doc = docs_with_metadata
#Adım 2: Embedding ve Vektör Veritabanı Oluşturma
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(docs_with_metadata, embedding_model)
print("\nVektör veritabanı oluşturuldu ve metinler yüklendi.")
metadata_field_info = [
    AttributeInfo(name="year", description="Dokümanın yayınlandığı yıl", type="integer"),
    AttributeInfo(name="topic", description="Dokümanın konusu, 'İK' veya 'Finans' olabilir", type="string"),
]
document_content_description = "Şirket politikaları ve raporları hakkında çeşitli bilgiler"

llm = ChatOpenAI(model="gpt-5-mini",temperature=0)

#Adım 3: Self-Querying Retriever kurulumu kullanımı
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents=document_content_description,  
    metadata_field_info=metadata_field_info,
    verbose=True
)


memory = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key="question",
    return_messages=True
)

#Template 
template = """Aşağıda verilen bağlam ve önceki konuşmalar ışığında, kullanıcının sorusuna cevap ver.
Eğer bağlam sorunun cevabını içermiyorsa, "Bilmiyorum" diye cevap ver.

Sohbet Geçmişi: {chat_history}

Bağlam: {context}

Kullanıcı: {question}
Asistan:"""

MY_PROMPT = PromptTemplate(   
    input_variables=["chat_history", "context", "input"],
    template=template
)

conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    combine_docs_chain_kwargs={"prompt": MY_PROMPT},
    memory=memory,
    retriever=self_query_retriever
)

print("\n--- Şirket İK Asistanı ---")
while True:
    query = input("Kullanıcı: ")
    if query.lower() in ["exit", "quit"]:
        print("Asistan: Görüşmek üzere!")
        break

    relevant_docs = vector_db.similarity_search(query, k=100)  # <-- k=5'ten 10'a çıkarıldı (tüm dokümanları al)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Zinciri çalıştır ve cevap al
    response = conversation.invoke({"question": query})  # <-- "input" yerine "question" kullanıldı, context kaldırıldı
    print("Asistan:", response["answer"])