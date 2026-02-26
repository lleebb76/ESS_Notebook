import streamlit as st
import PyPDF2
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

# Try to load the logo
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
    
    # The Multi-Model Toggle
    ai_choice = st.radio(
        "Select your AI Brain:",
        ("Gemini 1.5 Flash (Smart & Fast)", "Mistral 7B (Free Open Source)")
    )
    
    st.divider()
    
    st.header("📄 Your Sources")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Sources"):
        if uploaded_files:
            if not gemini_api_key:
                st.error("Gemini API key required to process embeddings.")
            else:
                with st.spinner("Reading and indexing your documents..."):
                    # 1. Read PDFs
                    raw_text = ""
                    for file in uploaded_files:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page in pdf_reader.pages:
                            # Basic error handling for unreadable PDF pages
                            text = page.extract_text()
                            if text:
                                raw_text += text + "\n"
                    
                    st.session_state.raw_text = raw_text

                    # 2. Split Text
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(raw_text)

                    # 3. Create Vector Store 
                    # We use Gemini for embeddings because it is fast, highly accurate, and cheap/free for this volume.
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    st.session_state.vector_store = vector_store
                    
                    st.success("Sources processed successfully!")
        else:
            st.warning("Please upload at least one PDF.")

# --- MAIN CHAT INTERFACE ---
st.divider()

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_question := st.chat_input("Ask a question about your sources..."):
    if st.session_state.vector_store is None:
        st.error("Please upload and process sources first!")
    elif ai_choice == "Mistral 7B (Free Open Source)" and not hf_api_token:
         st.error("Cannot use Mistral. Please add your Hugging Face API token to the environment variables.")
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate Response using RAG
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {ai_choice}..."):
                
                # --- DYNAMIC AI SELECTION ---
                if ai_choice == "Gemini 1.5 Flash (Smart & Fast)":
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
                else:
                    # Using Mistral 7B Instruct via Hugging Face's free inference API
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
                    
                    # Save to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
