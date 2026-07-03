from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not set in environment")

if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("MISTRAL_API_KEY not set in environment")


# def LLm_init(temperature: float = 0.1):
#     model = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=temperature,
#         max_tokens=None,
#         timeout=25,
#         max_retries=1,
#     )
#     return 

from langchain_mistralai import ChatMistralAI

def LLm_init(temperature: float = 0.1):
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=temperature,
        max_retries=1,
        timeout=25,
    )
