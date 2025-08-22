from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login
from dotenv import load_dotenv
import os
import sys

# 1️⃣ Load variables from .env if present
load_dotenv()

# 2️⃣ Get the token from env variable
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# 3️⃣ Check if token is loaded
if not token:
    print("❌ ERROR: Hugging Face API token not found.")
    print("Make sure you have either:")
    print(" - A .env file with: HUGGINGFACEHUB_ACCESS_TOKEN=your_token_here")
    print(" - Or set the token in your system environment variables")
    sys.exit(1)
else:
    print("✅ Hugging Face token loaded successfully.")

# 4️⃣ Log in to Hugging Face (stores token locally)
try:
    login(token=token)
    print("🔑 Logged in to Hugging Face successfully.")
except Exception as e:
    print("⚠️ Warning: Could not log in via huggingface_hub:", e)

# 5️⃣ Create HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Model repo
    task="text-generation",
    huggingfacehub_api_token=token  # still pass explicitly
)

# 6️⃣ Wrap it in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# 7️⃣ Run test prompt
try:
    result = model.invoke("What is the capital of England?")
    print("📝 Model Response:", result.content)
except Exception as e:
    print("❌ ERROR during model invocation:", e)
