from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """Introduction

LangChain is a powerful framework designed to simplify the development of applications powered by large language models (LLMs). One of its most important features is the ability to handle, process, and transform textual data from various sources. At the heart of this functionality lies the Document Loader.

A Document Loader is a component in LangChain responsible for fetching, parsing, and structuring documents from different sources such as PDFs, Word files, HTML pages, APIs, CSVs, and even proprietary knowledge bases. This makes it possible to provide unstructured data as input to downstream tasks such as summarization, question answering, retrieval-augmented generation (RAG), and semantic search.

Purpose of Document Loaders

The main goals of a Document Loader in LangChain are:

Data Ingestion: Extract information from various sources like files, websites, databases, or cloud storage.

Standardization: Convert unstructured or semi-structured data into a standardized format called a Document object.

Preprocessing: Allow optional cleaning and organization of the extracted data (e.g., removing metadata, splitting content).

Integration with Pipelines: Serve as the first step in workflows such as embedding creation, indexing, and retrieval.

"""


spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator='\n')

result = spliter.split_text(text)
#print(result)

loader = PyPDFLoader(r'C:\Users\anime\Downloads\LangChain\Machine_Learning\Combinations+With+Repetition.pdf')

docs = loader.lazy_load()

result2 = spliter.split_documents(docs)
print(result2[0].page_content)

