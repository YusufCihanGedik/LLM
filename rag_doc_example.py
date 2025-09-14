import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter # Daha akıllı splitter'a geçiyoruz
from langchain_community.document_loaders import PyPDFLoader # <-- YENİ BİLEŞEN: PDF Yükleyici

## API anahtarı .env dosyasından otomatik yüklenir

### Adım 1: PDF'den Veri Yükleme ve Parçalama 
# PDF dosyasını yüklemek için PyPDFLoader kullanıyoruz.
pdf_path ="doc/XYZteknoloji.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"PDF'den {len(documents)} belge yüklendi.")

#Genellikler her sayfa tek bir Document nesnesidir.

# print("İlk sayfanın içeriği:", documents[0].page_content[:300]) # içeriğin ilk 300 karakteri

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Metin {len(docs)} parçaya bölündü.")
print("İlk chunk:", docs[0].page_content)
### Adım 2: Embedding ve Vektör Veritabanı Oluşturma 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
##Alınacak veriye bağlı olarak from_texts veya from_documents kullanılabilir başka seçenekler de var.
vector_db = Chroma.from_documents(docs, embedding_model)
print("\nVektör veritabanı oluşturuldu ve metinler yüklendi.")

### Adım 3: Soru Sorma ve Cevap Üretme (OpenAI & .invoke() ile)
#gpt sadece stringden anlar query string olarak verilmeli
llm = ChatOpenAI(
    model="gpt-5-mini", # Örnek model ismi, mevcut en yeni modeli kullan.
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

print("\n--- Soru-Cevap (OpenAI ile) ---")

query = "XYZ Teknoloji'nin kuruluş yılı nedir?"
result = qa_chain.invoke({"query": query})

print("Soru:", query)
print("Cevap:", result['result'])

query_1 = "Yıllık izin hakkı ne kadar ve 5 yıldan fazla çalışanlar için durum nedir?"
result_1 = qa_chain.invoke({"query": query_1})

print("Soru:", query_1)
print("Cevap:", result_1['result'])

query_2 = "Hastalık izni için rapor ne zaman teslim edilmeli?"
result_2 = qa_chain.invoke({"query": query_2})
print("\nSoru:", query_2)
print("Cevap:", result_2['result'])