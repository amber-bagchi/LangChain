from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
import sys

# 1️⃣ Load variables from .env
load_dotenv()

# 2️⃣ Get token
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not token:
    print("❌ ERROR: Hugging Face API token not found.")
    sys.exit(1)
else:
    print("✅ Hugging Face token loaded successfully.")

# 3️⃣ Initialize API-based embeddings
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # you can switch to other models
    huggingfacehub_api_token=token
)

sample = """ The farmer woke up early to tend to his crops before the sun rose.  

Modern education systems are increasingly adopting online platforms for remote learning.  

The solar system consists of eight planets orbiting around the Sun.    

"""

text_splitter = SemanticChunker(
    embedding,
    breakpoint_threshold_type = 'standard_deviation',
    breakpoint_threshold_amount = 1
)

docs  = text_splitter.create_documents([sample])
print(len(docs))
print(docs[0].page_content)