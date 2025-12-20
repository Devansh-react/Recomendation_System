from langchain_google_genai import ChatGoogleGenerativeAI
import  os 
from dotenv import load_dotenv

load_dotenv()
os.environ


def LLm_init():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=2.0,  # Gemini 3.0+ defaults to 1.0
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    return model
    
    

if __name__ == "__main__":
    llm = LLm_init()
    response = llm.invoke("tell me abut python ?")
    print(response.content)


