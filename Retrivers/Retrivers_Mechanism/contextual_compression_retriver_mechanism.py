from langchain.schema import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
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
        page_content=(
            """The Grand Canyon is one of the most visited natural wonders in the world.
            Photosynthesis is the process by which green plants convert sunlight into energy.
            Millions of tourists travel to see it every year. The rocks date back millions of years."""
        ),
        metadata={"source": "Doc1"}
    ),
    Document(
        page_content=(
            """In medieval Europe, castles were built primarily for defense.
            The chlorophyll in plant cells captures sunlight during photosynthesis.
            Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
        ),
        metadata={"source": "Doc2"}
    ),
    Document(
        page_content=(
            """Basketball was invented by Dr. James Naismith in the late 19th century.
            It was originally played with a soccer ball and peach baskets.
            NBA is now a global league."""
        ),
        metadata={"source": "Doc3"}
    ),
    Document(
        page_content=(
            """The history of cinema began in the late 1800s. Silent films were the earliest form.
            Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
            Modern filmmaking involves complex CGI and sound design."""
        ),
        metadata={"source": "Doc4"}
    )
]

# Create vector store
vectorstore = FAISS.from_documents(docs, embedding)

# Create Base Retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Setup Compressor using llm
compressor = LLMChainExtractor.from_llm(model)

# Create Contextual Compression Retriever
contextual_compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
    verbose=True
)

# Query the retriever
query = "What is Photosynthesis?"

# Results
results = contextual_compression_retriever.invoke(query)

# Print results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")


