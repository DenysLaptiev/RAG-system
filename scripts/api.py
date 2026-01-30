import os
from dotenv import load_dotenv

load_dotenv()

openai_api = os.getenv("OPENAI_API_KEY")

if not openai_api:
    raise RuntimeError("OPENAI_API_KEY is not set")