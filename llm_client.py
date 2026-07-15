import os
from groq import Groq
from dotenv import load_dotenv

# read .env and load the api key into environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def safe_llm_call(messages, stream=False):
    try:
        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=stream
        )

        return response
    
    except Exception as e:
        if "rate_limit" in str(e).lower():
            raise Exception("rate_limit")
        raise e