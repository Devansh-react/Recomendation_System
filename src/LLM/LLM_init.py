from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not set in environment")


def LLm_init(temperature: float = 0.1):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_tokens=None,
        timeout=25,          # keep under your 30s hard cap, see note below
        max_retries=1,       # 2 retries + 60s timeout could blow the 30s budget
    )
    return model