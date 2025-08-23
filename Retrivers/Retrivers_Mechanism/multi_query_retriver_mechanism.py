from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
from huggingface_hub import login
import os
import sys

#  Load variables from .env if present
load_dotenv()

#  Get the token from env variable
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

#  Check if token is loaded
if not token:
    print(" ERROR: Hugging Face API token not found.")
    print("Make sure you have either:")
    print(" - A .env file with: HUGGINGFACEHUB_ACCESS_TOKEN=your_token_here")
    print(" - Or set the token in your system environment variables")
    sys.exit(1)
else:
    print(" Hugging Face token loaded successfully.")
    
#  Log in to Hugging Face (stores token locally)
try:
    login(token=token)
    print(" Logged in to Hugging Face successfully.")
except Exception as e:
    print(" Warning: Could not log in via huggingface_hub:", e)
    
# Create HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    
    repo_id="meta-llama/Llama-3.3-70B-Instruct",  # Model
    task="conversational",
    huggingfacehub_api_token=token   
)

# Create Model
model = ChatHuggingFace(llm=llm)

# Initialize API-based embeddings
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # you can switch to other models
    huggingfacehub_api_token=token
)

# Create documents

docs = [

    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"}
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"}
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"}
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"}
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"}
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"}
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"}
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"}
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"}
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"}
    ),
]

# Create VectorStore
vector_store = FAISS.from_documents(documents=docs, embedding= embedding)  # Create FAISS vector store

# Create Normal Retriever
similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Create Multi Query Retriever 
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=model  
)


# Query the retriever
query = "How to improve energy levels and maintain balance?"

# Results
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

# Print results
for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")

print("\n\n\n")

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")