import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever # <-- YENİ RETRIEVER
from langchain.chains.query_constructor.base import AttributeInfo # <-- Metadata tanımlamak için
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Document nesnesini manuel oluşturmak için
from langchain.schema import Document


docs_with_metadata = [
    Document(
        page_content="2023 İK Politikası: Tüm çalışanların yıllık 14 gün izni vardır. 5 yılı dolduranlar 20 gün alır.",
        metadata={"year": 2023, "doc_name": "IK_Politikasi_2023.pdf", "topic": "İK"}
    ),
    Document(
        page_content="2024 İK Politikası: Yeni politikaya göre, tüm çalışanların yıllık 16 gün izni vardır. 5 yılı dolduranlar ise 22 gün alır. Uzaktan çalışma haftada 3 güne çıkarılmıştır.",
        metadata={"year": 2024, "doc_name": "IK_Politikasi_2024.pdf", "topic": "İK"}
    ),
    Document(
        page_content="2023 Finans Raporu: Şirket karı %10 artmıştır.",
        metadata={"year": 2023, "doc_name": "Finans_Raporu_2023.pdf", "topic": "Finans"}
    ),
    Document(
        page_content="2024 Finans Raporu: Yeni yatırımlar sayesinde şirket karı %15 artmıştır.",
        metadata={"year": 2024, "doc_name": "Finans_Raporu_2024.pdf", "topic": "Finans"}
    )
]

# docs = text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc = docs_with_metadata

#adım 2: Embedding ve Vektör Veritabanı Oluşturma
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(docs_with_metadata, embedding_model)
print("\nVektör veritabanı oluşturuldu ve metinler yüklendi.")

#adım 3: Self-Querying Retriever kurulumu kullanımı
metadata_fields_info = [
    AttributeInfo(
        name="year",
        description="Belgenin yayınlandığı yıl",
        type="integer"
    ),
    AttributeInfo(
        name="doc_name",
        description="Belgenin dosya adı",
        type="string"
    ),
    AttributeInfo(
        name="topic",
        description="Belgenin konusu, örn. İK, Finans, Teknik",
        type="string"
    )
]
document_content_description = "Şirket politikaları ve finans raporları"

llm = ChatOpenAI(
    model="gpt-5-mini", # Örnek model ismi, mevcut en
    temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    metadata_field_info=metadata_fields_info,
    document_contents=document_content_description,
    verbose=True # Arka planda neler olduğunu görmek için Verbose True Yapılabilir
)

query = "2024 yılına ait İK politikası nedir?"
"""invoke, retriever'ın LLM ile birlikte sorguyu işleyip, en uygun dökümanları döndürmesini sağlar.
Sorgu doğal dilde yazılır; retriever bunu analiz edip, vektör veritabanında arama ve filtreleme yapar.
Sonuç olarak, ilgili dökümanlar (Document nesneleri) bir liste olarak döner.
Yani, invoke burada "sorguyu çalıştır ve sonuçları getir" anlamına gelir."""
docs = retriever.invoke(query)
print(f"\nSoru: {query}")
print(f"{len(docs)} belge getirildi.")
for doc in docs:
    print(f"- İçerik: {doc.page_content}")
    print(f"  - Metadata: {doc.metadata}")
    
    
#Arka planda :
#query='uzaktan çalışma' filter=Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='year', value=2024), 
#Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='topic', value='İK') limit=None
#Burada yıl ve konuya göre filtreleme yapıldı.