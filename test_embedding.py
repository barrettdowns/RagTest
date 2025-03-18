import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
)

try:
    print("Testing embedding with model:", os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))
    embeddings = client.embeddings.create(
        model=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        input="Test embedding model"
    )
    print("Success! Embedding dimensions:", len(embeddings.data[0].embedding))
except Exception as e:
    print("Error:", e)
