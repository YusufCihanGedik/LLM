import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI 
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter


### Adım 1: Veri Yükleme ve Parçalama (DEĞİŞİKLİK YOK)
company_policy_text = """
İK Politikaları Belgesi
1. Yıllık İzin Hakkı: Şirketimizde tam zamanlı çalışan her personel, bir tam yılı doldurduğunda 14 iş günü yıllık ücretli izin hakkı kazanır. 5 yıldan fazla kıdeme sahip çalışanlar için bu süre 20 iş günüdür. Yıllık izinler, bir sonraki yıla devredilemez. İzin talepleri, planlanan tarihten en az 2 hafta önce İK departmanına bildirilmelidir.
2. Hastalık İzni: Çalışanlar, doktor raporu sunmaları koşuluyla yılda 10 güne kadar ücretli hastalık izni kullanabilirler. Raporun 2 gün içinde İK'ya teslim edilmesi zorunludur.
3. Uzaktan Çalışma Politikası: Mühendislik ve ürün departmanları, yöneticilerinin onayıyla haftada 2 güne kadar uzaktan çalışabilir. Uzaktan çalışma günleri takım içinde koordine edilmelidir.
"""

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_text(company_policy_text)
print(f"Metin {len(docs)} parçaya bölündü.")


### Adım 2: Embedding ve Vektör Veritabanı Oluşturma (DEĞİŞİKLİK YOK)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_texts(docs, embedding_model)
print("\nVektör veritabanı oluşturuldu ve metinler yüklendi.")


### Adım 3: Soru Sorma ve Cevap Üretme (OpenAI & .invoke() ile)

# OpenAI LLM'ini başlatıyoruz. temperature=0, modelin daha deterministik ve olgusal cevaplar vermesini sağlar.
llm = ChatOpenAI(
    model="gpt-5-mini", # Örnek model ismi, mevcut en yeni modeli kullan.
    temperature=0)

# Standart RetrievalQA zincirimizi kuruyoruz.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=vector_db.as_retriever()
)

print("\n--- Soru-Cevap (OpenAI ile) ---")

query = "5 yıldan fazla çalışan birinin yıllık izin hakkı ne kadar?"

# LangChain'in yeni standartı olan .invoke() metodunu kullanıyoruz.
# Bu metod, girdi olarak bir dictionary alır.
result = qa_chain.invoke({"query": query})

print("Soru:", query)
# .invoke() çıktısı da bir dictionary'dir. Cevap 'result' anahtarının altındadır.
print("Cevap:", result['result'])

query_2 = "Mühendisler evden çalışabilir mi?"
result_2 = qa_chain.invoke({"query": query_2})
print("\nSoru:", query_2)
print("Cevap:", result_2['result'])