
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq.chat_models import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()
model=ChatGroq(model="llama-3.3-70b-versatile")
loader=WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
raw_documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(raw_documents)

#embeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectore_store=FAISS.from_documents(documents,embeddings)
retriever=vectore_store.as_retriever()

rag_template="""You are a senior developer. Answer the question strictly using the provided context.
Context: {context}
Question: {question}"""

rag_prompt=ChatPromptTemplate.from_template(rag_template)

plain_prompt=ChatPromptTemplate.from_template("Answer this question based on your general knowledge: {question}")


#without rag
def without_rag(query):
    chain = plain_prompt | model | StrOutputParser()
    return chain.invoke(query)

#with rag
def with_rag(query):
    chain=(
        {"context":retriever,"question":RunnablePassthrough()} |
        rag_prompt |
        model |
        StrOutputParser()
    )
    return chain.invoke(query)


def start():
    while True:
        query=input(f"Query...")
        if query=="exit":
            break
        print("Without RAG")
        print(without_rag(query))
        print("-"*100)
        print("With RAG")
        print(with_rag(query))

if __name__=="__main__":
    start()