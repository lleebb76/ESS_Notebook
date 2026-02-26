import streamlit as st
import PyPDF2
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE SETUP ---
st.set_page_config(page_title="ESS Notebook", page_icon="📚", layout="wide")

# Try to load the logo
try:
    st.image("logo.png", width=400)
except FileNotFoundError:
    st.title("📚 ESS Notebook")

st.markdown("### Your AI-powered research assistant for Environmental Standards.")

# --- API KEY CHECK ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.warning("⚠️ Please set your GEMINI_API_KEY in the environment variables.")
    st.stop()

# --- INITIALIZE SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

# --- SIDEBAR: SOURCE UPLOAD ---
with st.sidebar:
    st.header("📄 Your Sources")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Sources"):
        if uploaded_files:
            with st.spinner("Reading and indexing your documents..."):
                # 1. Read PDFs
                raw_text = ""
                for file in uploaded_files:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text() + "\n"
                
                st.session_state.raw_text = raw_text

                # 2. Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(raw_text)

                # 3. Create Vector Store (Embeddings)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.session_state.vector_store = vector_store
                
                st.success("Sources processed successfully!")
        else:
            st.warning("Please upload at least one PDF.")

    # Summary Button
    if st.session_state.raw_text and st.button("Generate Source Summary"):
        with st.spinner("Summarizing sources..."):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            prompt = f"Provide a comprehensive but concise summary of the following text:\n\n{st.session_state.raw_text[:15000]}..." # Truncated for token limits on basic summaries
            response = llm.invoke(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": f"**Source Summary:**\n{response.content}"})

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
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate Response using RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                
                system_prompt = (
                    "You are ESS Notebook, an expert AI assistant for Environmental Standards Scotland. "
                    "Use the provided retrieved context to answer the user's question. "
                    "If you don't know the answer based on the context, just say that you don't know. "
                    "Context: {context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": user_question})
                answer = response["answer"]
                st.markdown(answer)
                
        # Save to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
