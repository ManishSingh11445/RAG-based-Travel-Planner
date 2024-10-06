import os
import streamlit as st
import pickle
import time
import langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


st.title("Travel Planner")
st.sidebar.title("Blog urls")

main_placeholder = st.empty()
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL: {i + 1}")
    urls.append(url)

embeddings = HuggingFaceEmbeddings()
file_path = "vector_index.pkl"
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(repo_id=repo_id,huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,temperature=0.1)
prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1.If you don't know the answer ,don't try to make up an answer. Just say 'I can't find the final answer but you may want to check the following links.'
2. If you find the answer, write the answer in a concise way with three sentences maximum.
3. Try to keep the answer concise and to the point.

{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])

process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Strated...")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    main_placeholder.text("Splitting the document...")
    docs = text_splitter.split_documents(data)

    vectorindex = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Started building Vector Embeddings")
    time.sleep(2)
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex,f)
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        main_placeholder.text("Vector DB file_path path exists...")
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
        retriever = vectorIndex.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        main_placeholder.text("Applying retrievalQA...")
        retrievalQA = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt":PROMPT}
        )

        result = retrievalQA.invoke({"query": query})
        if result:
            main_placeholder.text("Result generated...")
            st.header("Answer: ")
            st.subheader(result['result'])
        else:
            main_placeholder.text("No answer...")















