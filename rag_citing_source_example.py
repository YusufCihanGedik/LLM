#Kaynak Gösterme ile RAG Örneği
#Bu örnek, bir PDF belgesinden veri yükleyip, parçalayarak,
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# kaynak göstererek cevap veren bir RAG sistemi kurar.
from langchain.chains import RetrievalQAWithSourcesChain # <-- DOĞRU ZİNCİRİ İMPORT ET
#Aşşağıdaki kaynak göstermek için yetersiz
# from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter # Daha akıllı splitter'a geç
from langchain_community.document_loaders import PyPDFLoader # <-- YENİ BİLEŞEN: PDF Yükleyici

## API anahtarı .env dosyasından otomatik yüklenir

### Adım 1: PDF'den Veri Yükleme ve Parçalama (YENİ)

pdf_path ="doc/XYZteknoloji.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"PDF'den {len(documents)} belge yüklendi.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Metin {len(docs)} parçaya bölündü.")
print("İlk chunk:", docs[0].page_content)

### Adım 2: Embedding ve Vektör Veritabanı Oluşturma (DEĞİŞİKLİK YOK)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(docs, embedding_model)
print("\nVektör veritabanı oluşturuldu ve metinler yüklendi.")

### Adım 3: Soru Sorma ve Cevap Üretme (OpenAI & .invoke() ile)
llm =ChatOpenAI(
    model="gpt-5-mini", # Örnek model ismi, mevcut en yeni modeli kullan.
)

qa_chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever()
)

print("\n--- Soru-Cevap (Kaynak Göstererek) ---")

query = "5 yıldan fazla çalışanların yıllık izin hakkı nedir?"
# Bu zincir, girdi olarak 'question' anahtarını bekler.
result = qa_chain.invoke({"question": query}) 

print("Soru:", query)
# Bu zincirin çıktısında 'answer' ve 'sources' anahtarları bulunur.
print("\nCevap:", result['answer']) 
print("Kaynaklar:", result['sources'])

# Başka bir örnek
query_2 = "Hastalık izni için rapor ne zaman teslim edilmeli?"
result_2 = qa_chain.invoke({"question": query_2})

print("\nSoru:", query_2)
print("\nCevap:", result_2['answer'])
print("Kaynaklar:", result_2['sources'])