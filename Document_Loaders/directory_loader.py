from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = r'C:\Users\anime\Downloads\LangChain\Machine_Learning',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)