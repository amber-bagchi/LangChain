from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
import os
import sys



loader = TextLoader(r"C:\Users\anime\Downloads\LangChain\data.txt", encoding = 'utf-8')

docs = loader.load()

#print(docs)

#print(type(docs))

#print(docs[0])

#print(type(docs[0]))

#print(docs[0].page_content)

#print(docs[0].metadata)


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
    
    repo_id="meta-llama/Llama-3.3-70B-Instruct",  # Model
    task="text-generation",
    huggingfacehub_api_token=token   
)

# 6️⃣ Create Model
model = ChatHuggingFace(llm=llm)

# 7️⃣ Create Parser
parser = StrOutputParser()

# 8️⃣ Create Prompt
prompt = PromptTemplate(
    template="Write a summary of the following report \n {report}",
    input_variables=["report"]
)

# 9️⃣ Create Chain
chain = prompt | model | parser

print(chain.invoke({"report": docs[0].page_content}))