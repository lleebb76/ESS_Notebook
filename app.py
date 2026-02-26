import streamlit as st
import PyPDF2
import docx2txt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE SETUP ---
st.set_page_config(page_title="ESS Notebook", page_icon="📚", layout="wide")

try:
    st.image("logo.png", width=400)
except FileNotFoundError:
    st.title("📚 ESS Notebook")

st.markdown("### Your AI-powered research assistant for Environmental Standards.")

# --- API KEY CHECKS ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")
hf_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not gemini_api_key:
    st.warning("⚠️ Please set your GEMINI_API_KEY in the environment variables.")
if not hf_api_token:
    st.info("ℹ️ HUGGINGFACEHUB_API_TOKEN not found. The free AI option will be disabled.")

# --- INITIALIZE SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

# --- SIDEBAR: CONTROLS & UPLOAD ---
with st.sidebar:
    st.header("⚙️ AI Settings")
    ai_choice = st.radio(
        "Select your AI Brain:",
        ("Gemini 2.5 Flash (Smart & Fast)", "Mistral 7B (Free Open Source)")
    )
    
    st.divider()
    
    st.header("📄 Your Sources")
    
    # Expanded File Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "docx", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )
    
    # Web Link Input
    web_links = st.text_area("Add Web Links (paste one URL per line)")
    
    if st.button("Process Sources"):
        if uploaded_files or web_links.strip():
            if not gemini_api_key:
                st.error("Gemini API key required to process embeddings.")
            else:
                with st.spinner("Reading and indexing your sources..."):
                    raw_text = ""
                    
                    # 1. Process Uploaded Files
                    if uploaded_files:
                        for file in uploaded_files:
                            filename = file.name.lower()
                            try:
                                if filename.endswith(".pdf"):
                                    pdf_reader = PyPDF2.PdfReader(file)
                                    for page in pdf_reader.pages:
                                        text = page.extract_text()
                                        if text: raw_text += text + "\n"
                                elif filename.endswith(".docx"):
                                    raw_text += docx2txt.process(file) + "\n"
                                elif filename.endswith(".csv"):
                                    df = pd.read_csv(file)
                                    raw_text += df.to_string() + "\n"
                                elif filename.endswith(".xlsx"):
                                    df = pd.read_excel(file)
                                    raw_text += df.to_string() + "\n"
                                elif filename.endswith(".txt"):
                                    raw_text += file.getvalue().decode("utf-8") + "\n"
                            except Exception as e:
                                st.error(f"Could not read {file.name}: {e}")
                    
                    # 2. Process Web Links
                    if web_links.strip():
                        urls = [url.strip() for url in web_links.split('\n') if url.strip()]
                        for url in urls:
                            try:
                                response = requests.get(url, timeout=10)
                                soup = BeautifulSoup(response.content, 'html.parser')
                                raw_text += ' '.join(soup.stripped_strings) + "\n"
                            except Exception as e:
                                 st.error(f"Could not read URL {url}: {e}")
                    
                    if raw_text.strip():
                        st.session_state.raw_text = raw_text

                        # Split Text
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_text(raw_text)

                        # Create Vector Store using the NEW Gemini Embedding Model
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/gemini-embedding-001", 
                            google_api_key=gemini_api_key
                        )
                        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                        st.session_state.vector_store = vector_store
                        
                        st.success("Sources processed successfully!")
                    else:
                        st.error("No readable text was found in the provided sources.")
        else:
            st.warning("Please upload a file or enter a web link.")

# --- MAIN CHAT INTERFACE ---
st.divider()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question about your sources..."):
    if st.session_state.vector_store is None:
        st.error("Please process your sources first!")
    elif ai_choice == "Mistral 7B (Free Open Source)" and not hf_api_token:
         st.error("Cannot use Mistral. Please add your Hugging Face API token to the environment variables.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {ai_choice}..."):
                
                if ai_choice == "Gemini 2.5 Flash (Smart & Fast)":
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
                else:
                    llm = HuggingFaceEndpoint(
                        repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
                        huggingfacehub_api_token=hf_api_token,
                        temperature=0.3,
                        max_new_tokens=512
                    )
                
                system_prompt = (
                    "You are ESS Notebook, an expert AI assistant for Environmental Standards Scotland. "
                    "Use the provided retrieved context to answer the user's question clearly and professionally. "
                    "If you don't know the answer based on the context, just say that you don't know. "
                    "Context: {context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                try:
                    response = rag_chain.invoke({"input": user_question})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
