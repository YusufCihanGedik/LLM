# Chat Memory Örneği
# Bu örnek, bir PDF belgesinden veri yükleyip, parçalayarak,
# vektör veritabanına kaydeder ve ardından OpenAI LLM kullanarak
# bu veritabanına dayalı sorulara cevap verir.
# .env dosyasından API anahtarı yüklenir
from dotenv import load_dotenv
load_dotenv()

import os 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain # <-- YENİ ZİNCİR
from langchain.memory import ConversationBufferMemory # <-- YENİ BİLEŞEN: Konuşma Belleği
from langchain_text_splitters import RecursiveCharacterTextSplitter # Daha akıllı splitter'a geçiyoruz
from langchain_community.document_loaders import PyPDFLoader # <-- YENİ BİLEŞEN: PDF Yükleyici

## API anahtarı .env dosyasından otomatik yüklenir

### Adım 1: PDF'den Veri Yükleme ve Parçalama 
pdf_path ="doc/XYZteknoloji.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"PDF'den {len(documents)} belge yüklendi.")

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

llm = ChatOpenAI(
    model="gpt-5-mini"
)
# Bellek nesnesini oluşturuyoruz. Bu, konuşmaları saklayacak olan bileşendir.
# `memory_key="chat_history"` -> Zincirin belleği bu isimle tanımasını sağlar.
# `return_messages=True` -> Mesajları özel bir formatta saklar.

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory
)

print("\n--- Sohbet Başladı ---")

# İlk soru
query_1 = "5 yıldan fazla çalışan birinin yıllık izin hakkı ne kadar?"
result_1 = conversation_chain.invoke({"question": query_1})
print("Kullanıcı:", query_1)
print("Asistan:", result_1['answer'])

# İkinci, takip sorusu 
# "Bu izin" ifadesi, bir önceki konuşmaya referans veriyor.
query_2 = "Bu izin için ne zaman talep oluşturmalıyım?"
result_2 = conversation_chain.invoke({"question": query_2})
print("\nKullanıcı:", query_2)
print("Asistan:", result_2['answer'])

# Hafızanın içeriğini görelim 
print("\n--- Hafıza İçeriği ---")
print(memory.buffer)