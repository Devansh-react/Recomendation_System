from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("MISTRAL_API_KEY not set in environment")


def LLm_init(temperature: float = 0.1):
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=temperature,
        max_retries=1,
        timeout=25,
    )