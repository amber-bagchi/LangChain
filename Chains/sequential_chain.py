from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
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
    
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Model
    task="text-generation",
    huggingfacehub_api_token=token   
)

# 6️⃣ Create Model
model = ChatHuggingFace(llm=llm)

# 7️⃣ Create Prompt
prompt = PromptTemplate(
    template="Generate a detailed report about the  {subject}",
    input_variables=["subject"]    
)

# 8️⃣ Create Output Parser
output_parser = StrOutputParser()

# 9️⃣ Create Prompt 2
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from thr following text \n {text}",
    input_variables=["text"]    
)

# 10️⃣ Create Chain
chain = prompt | model | output_parser | prompt2 | model | output_parser

# 11️⃣ Invoke Chain
result = chain.invoke({"subject": "Celestial Creatures"})

print(result)
chain.get_graph().print_ascii()
 