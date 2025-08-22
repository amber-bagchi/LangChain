from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field
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

# 7️⃣ Create a Pydantic Output Parser
class Person(BaseModel):
    name: str = Field(description='The name of the person')
    age: int = Field(gt = 18,description='The age of the person')
    city: str = Field(description='Name of city of the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate the name, age, and city of a fictional {place} person. "
        "Return ONLY valid JSON, nothing else. \n"
        "{format_instruction}"
    ),
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'place': 'Indian'}) 
result = model.invoke(prompt) 
final_result = parser.parse(result.content) 
print(final_result)