from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
import os
import sys

# 1️⃣ Load variables from .env if present
load_dotenv()

# 2️⃣ Get the token from env variable
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# 3️⃣ Check if token is loaded
if not token:
    print("❌ ERROR: Hugging Face API token not found.")
    sys.exit(1)
else:
    print("✅ Hugging Face token loaded successfully.")

# 4️⃣ Log in to Hugging Face
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

# Prompt for joke
prompt = PromptTemplate(
    template='Write a creative joke about the {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

# Chain for joke
joke_chain = RunnableSequence(prompt, model, parser)

paralell_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(
        lambda x: len(x.split())
    )
})

final_chain = RunnableSequence(joke_chain, paralell_chain)

result = final_chain.invoke({'topic': 'Education'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)