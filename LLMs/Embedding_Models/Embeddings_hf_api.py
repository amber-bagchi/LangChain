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

# 4️⃣ Example documents
docs = [
    "This is a sentence",
    "This is another sentence",
    "This is a third"
]

# 5️⃣ Generate embeddings
try:
    embeddings = embedding.embed_documents(docs)
    print("✅ Embeddings generated successfully\n")

    for i, emb in enumerate(embeddings):
        print(f"🔹 Document {i+1}: {docs[i]}")
        print(f"   Vector length: {len(emb)}")
        print(f"   First 10 values: {emb[:10]} ...\n")  # show first 10 values only
except Exception as e:
    print("❌ ERROR during embedding generation:", e)
