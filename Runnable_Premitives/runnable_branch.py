from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableBranch
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
llm1 = HuggingFaceEndpoint(
    
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Model
    task="text-generation",
    huggingfacehub_api_token=token   
)

# 6️⃣ Create Model
model1 = ChatHuggingFace(llm=llm1)


prompt1 = PromptTemplate(
    template='Write an detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text under 50 words \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model1, parser)

summarize_chain = RunnableSequence(prompt2, model1, parser)

branch_chain = RunnableBranch(

    (lambda x: len(x.split()) > 200, summarize_chain),
    RunnablePassthrough()
)

final_chain  = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))


      