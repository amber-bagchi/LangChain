from langchain_community.retrievers import WikipediaRetriever

# Initialize the WikipediaRetriever
retriever = WikipediaRetriever(top_k_results=5, lang="en")

# Query the retriever
query = "The geopolitical history of Iran from the perspective of India"

# Retrieve documents from Wikipedia
documents = retriever.invoke(query)

# Print the retrieved documents
for i, doc in enumerate(documents):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")
    

