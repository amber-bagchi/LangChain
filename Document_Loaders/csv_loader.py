from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = r'C:\Users\anime\Downloads\LangChain\Machine_Learning\Iris.csv')
data = loader.load()

print(data[0])

print(len(data))

