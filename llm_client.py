"""
llm_client.py

Initialises the Groq API client and provides a safe wrapper for LLM API calls
with rate limit detection and error handling.
"""

import os
from groq import Groq
from dotenv import load_dotenv

# read .env and load the api key into environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def safe_llm_call(messages, stream=False):

    """
    Wrapper for Groq API calls with error handling.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        stream: If True, returns a streaming response, defaults to False

    Returns:
        Groq API response object

    Raises:
        Exception('rate_limit'): If Groq rate limit is hit
        Exception: For any other API errors
    """
        
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