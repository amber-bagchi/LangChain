from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = r'C:\Users\anime\Downloads\LangChain\Machine_Learning',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

print(len(docs))

for doc in docs:
    print(doc.metadata)

# Loads One doc or page at a time faster and memory efficient