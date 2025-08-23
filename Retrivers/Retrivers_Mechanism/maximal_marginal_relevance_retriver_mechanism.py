from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

# Create a list of documents
docs = [
    Document(page_content="LangChain provides an MMR retriever to return results that are both relevant and diverse.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="The MMR retriever in LangChain reduces redundancy by penalizing documents that are too similar.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="Maximal Marginal Relevance is a method to select documents that balance query relevance and diversity.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="In LangChain, MMR retriever is often used in semantic search pipelines.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="LangChain’s retrievers allow developers to choose between similarity search and MMR search.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="The MMR retriever ensures that not all retrieved chunks are near-duplicates of each other.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="LangChain integrates HuggingFace embeddings with MMR retriever for better semantic retrieval.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="MMR is useful when dealing with long documents split into multiple chunks.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="LangChain retrievers can be combined with vector stores like FAISS or Chroma.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="Developers use MMR retriever when they want answers with more coverage of the topic.",
             metadata={"topic": "LangChain-MMR"}),

    # 🔁 Some intentionally similar
    Document(page_content="MMR retriever in LangChain reduces duplication in search results.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="LangChain’s MMR retriever balances relevance with novelty of results.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="The retriever is important in RAG pipelines where diverse contexts improve answers.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="LangChain supports both similarity search and MMR search for retrievers.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="MMR is especially useful for chatbots that require varied knowledge snippets.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="In LangChain, retrievers can be customized with similarity or MMR as retrieval modes.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="LangChain makes it easy to configure retrievers with embeddings and vector databases.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="Maximal Marginal Relevance reduces redundancy in retrieved document sets.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="MMR retriever improves coverage of answers by including diverse information sources.",
             metadata={"topic": "LangChain-MMR"}),

    Document(page_content="LangChain pipelines benefit from MMR when summarizing or answering questions with multiple perspectives.",
             metadata={"topic": "LangChain-MMR"}),
]

# Load variables from .env
load_dotenv()

# 2️⃣ Get token
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Initialize API-based embeddings
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # you can switch to other models
    huggingfacehub_api_token=token
)

# Create a vector store FAISS
vector_store = FAISS.from_documents(documents=docs, embedding= embedding)

# Enable MMR in the retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 1})

# Query the retriever
query = "What is Langchain?"

# Results
result = retriever.invoke(query)

# Print results
for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")
    print(f"Metadata:\n{doc.metadata}")
    
    
#  MMR (Maximal Marginal Relevance) Retriever Mechanism
# MMR is a retrieval technique that balances two things:
# 1. Relevance → Documents that are most related to the query.
# 2. Diversity  → Documents that add new information instead of being duplicates.
#
# Instead of just retrieving the top-k most similar chunks, MMR ensures the results
# cover a wider range of perspectives. This reduces redundancy and improves coverage.
#
# Use Case:
# MMR is especially useful in RAG (Retrieval Augmented Generation) pipelines,
# chatbots, Q&A systems, and summarization tasks where diverse context improves 
# the final response quality.