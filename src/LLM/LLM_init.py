from langchain_google_genai import ChatGoogleGenerativeAI
import  os 
from dotenv import load_dotenv

load_dotenv()
os.environ


def LLm_init():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=2.0,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )
    
    return model

if __name__=="__main__":
    model = LLm_init()
    model.invoke("hello")
    
    