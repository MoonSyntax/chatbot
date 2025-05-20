import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Interactive RAG-based LLM for Multi-PDF Document Analysis", divider='rainbow')

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()


# test_1000_tokens.py

import math
import random
from typing import List, Dict

class DataProcessor:
    def __init__(self, data: List[int]):
        self.data = data

    def clean_data(self) -> List[int]:
        return [x for x in self.data if x is not None and isinstance(x, int)]

    def scale_data(self, factor: float) -> List[float]:
        return [x * factor for x in self.clean_data()]

    def normalize_data(self) -> List[float]:
        clean = self.clean_data()
        min_val = min(clean)
        max_val = max(clean)
        return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in clean]

    def compute_statistics(self) -> Dict[str, float]:
        clean = self.clean_data()
        return {
            "mean": sum(clean) / len(clean),
            "min": min(clean),
            "max": max(clean),
            "std_dev": math.sqrt(sum((x - sum(clean)/len(clean))**2 for x in clean) / len(clean))
        }

def generate_data(n: int) -> List[int]:
    return [random.randint(1, 100) for _ in range(n)]

def main():
    data = generate_data(100)
    processor = DataProcessor(data)

    print("Original Data:", data[:10])
    print("Clean Data:", processor.clean_data()[:10])
    print("Scaled Data (x2):", processor.scale_data(2.0)[:10])
    print("Normalized Data:", processor.normalize_data()[:10])
    print("Statistics:", processor.compute_statistics())

