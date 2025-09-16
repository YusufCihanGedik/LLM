import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain # <-- YENİ ZİNCİR
from langchain.memory import ConversationBufferMemory # <-- YENİ BİLEŞEN: Konuşma Belleği
from langchain_text_splitters import RecursiveCharacterTextSplitter # Daha akıllı splitter'a geçiyoruz
from langchain_community.document_loaders import PyPDFLoader # <-- YENİ BİLEŞEN: PDF Yükleyici
from langchain.retrievers.self_query.base import SelfQueryRetriever # <-- YENİ RETRIEVER
from langchain.chains.query_constructor.base import AttributeInfo # <-- Metadata tanımlamak için
from langchain.schema import Document #Document nesnelerini manuel oluşturmak için 

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
    )
    Document(
        page_content="Jon Doe eşi Jane Doe ile birlikte 2024 yılında şirkete katıldı.",
        metadata={"year": 2024, "topic": "Finans"}
    )
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

#Adım 3: Self-Querying Retriever kurulumu kullanımı
# 3.1. Self-Querying Retriever'ı oluşturuyoruz (Önceki kodla aynı)
metadata_field_info = [
    AttributeInfo(name="year", description="Dokümanın yayınlandığı yıl", type="integer"),
    AttributeInfo(name="topic", description="Dokümanın konusu, 'İK' veya 'Finans' olabilir", type="string"),
]
document_content_description = "Şirket politikaları ve raporları hakkında çeşitli bilgiler"

llm = ChatOpenAI(model="gpt-5-mini",temperature=0)

# SelfQueryRetriever'ı oluşturuyoruz.
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    metadata_field_info=metadata_field_info,
    document_contents=document_content_description,
    verbose=True # Arka planda neler olduğunu görmek için Verbose True Yapılabilir
)
# 3.2. Sohbet Hafızasının tutumlası gerekiyor 
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 3.3. ConversationalRetrievalChain'i KURUYORUZ 
# ...retriever olarak standart olan yerine KENDİ SELF-QUERY RETRIEVER'ımızı veriyoruz!
#ConversationalRetrievalChain SelfQueryRetriever faydalanarak yapısal soruları anlamayı destekler
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=self_query_retriever,
    memory=memory
)

print("\n--- Sohbet Başladı ---")

query_1 = "2024 yılındaki finans raporunda ne gibi gelişmeler var?"
result_1 = conversation_chain.invoke({"question": query_1})
print("Kullanıcı:", query_1)
print("Asistan:", result_1['answer'])

query_2 = "Peki bir önceki yıla göre durum nasıldı?"
result_2 = conversation_chain.invoke({"question": query_2})
print("\nKullanıcı:", query_2)
print("Asistan:", result_2['answer'])