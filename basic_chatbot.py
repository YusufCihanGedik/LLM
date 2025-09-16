import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory # <-- YENİ BİLEŞEN: Konuşma Belleği

#Direk llm sormak istiyorsak doc ve emmbeding gerekmez
llm = ChatOpenAI(
    model="gpt-5-mini"
)

memory = ConversationBufferMemory(

)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False # True yaparsanız işlem adımlarını görebilirsiniz
)

def main():
    print("\n--- Basit Chatbot Başladı ---")
    while True:
        user_input = input("Kullanıcı: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot kapatılıyor.")
            break
        result = conversation.invoke({"question": user_input})
        print("Asistan:", result['answer'])

if __name__ == "__main__":
    main()