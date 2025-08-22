from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
import os
import sys



loader = PyPDFLoader(r"C:\Users\anime\Downloads\LangChain\Machine_Learning\LLM_Reseach_Paper.pdf")

docs = loader.load()

# print(docs)

print(len(docs))

print(docs[0].page_content)

print(docs[1].metadata)