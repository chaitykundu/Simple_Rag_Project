import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# 🛠️ Patch missing event loop issue in Streamlit thread
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 🔐 Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize LLM and Embedding with API key
llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Streamlit UI
st.set_page_config(page_title="PDF Viewer", layout="wide")

st.title("🔍 My RAG PDF Assistant")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        with pdfplumber.open(uploaded_file) as pdf:
            raw_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents([raw_text])

        # Vectorstore
        if not os.path.exists("vectorstore"):
            os.makedirs("vectorstore")
        vectorstore = FAISS.from_documents(chunks, embedding)
        vectorstore.save_local("vectorstore/faiss_index")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # User question
        st.success("✅ PDF processed. Ask a question below.")
        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(query)
                st.subheader("📘 Answer:")
                st.write(response["result"])

                with st.expander("🧾 Sources"):
                    for doc in response["source_documents"]:
                        st.markdown(f"**Page snippet:** {doc.page_content[:400]}...")
