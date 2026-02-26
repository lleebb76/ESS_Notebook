import streamlit as st
import PyPDF2
import docx2txt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
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
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

if not gemini_api_key:
    st.warning("⚠️ Please set your GEMINI_API_KEY in the environment variables.")
if not hf_api_token:
    st.error("🛑 HUGGINGFACEHUB_API_TOKEN is missing. This is required to process documents for free.")
if not anthropic_api_key:
    st.info("ℹ️ ANTHROPIC_API_KEY not found. The Claude option will be disabled.")

# --- INITIALIZE SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "sources_processed" not in st.session_state:
    st.session_state.sources_processed = False

# --- SIDEBAR: CONTROLS & UPLOAD ---
with st.sidebar:
    st.header("⚙️ AI Settings")
    ai_choice = st.radio(
        "Select your AI Brain:",
        ("Gemini 2.5 Flash (Smart & Fast)", "Claude 3.5 Sonnet (Nuanced & Logical)", "Mistral 7B (Free Open Source)")
    )
    
    # --- NEW KNOWLEDGE BASE TOGGLE ---
    knowledge_source = st.radio(
        "Knowledge Base:",
        ("Strictly Uploaded Sources", "Include General Knowledge")
    )
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    
    st.header("📄 Your Sources")
    st.info("Upload all files and paste all links before clicking Process.")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "docx", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )
    
    web_links = st.text_area("Add Web Links (paste one URL per line)")
    
    # STEP 1: Process Sources
    if st.button("1. Process & Index Sources"):
        if uploaded_files or web_links.strip():
            with st.spinner("Reading and indexing your sources..."):
                raw_text = ""
                
                # Process Uploaded Files
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
                
                # Process Web Links
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

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    chunks = text_splitter.split_text(raw_text)

                    # EMBEDDING STRATEGY
                    if hf_api_token:
                        try:
                            embeddings = HuggingFaceEndpointEmbeddings(
                                model="sentence-transformers/all-MiniLM-L6-v2",
                                task="feature-extraction",
                                huggingfacehub_api_token=hf_api_token
                            )
                            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                            st.session_state.vector_store = vector_store
                            st.session_state.sources_processed = True
                            st.success("Sources successfully indexed! You can now generate an overview or start asking questions.")
                        except Exception as e:
                            st.error(f"Hugging Face Embedding Error: {e}")
                    else:
                        st.error("Hugging Face API token missing. Cannot process.")
                else:
                    st.error("No readable text was found in the provided sources.")
        else:
            st.warning("Please upload a file or enter a web link.")

    # STEP 2: Explicit Summary Generation
    if st.session_state.sources_processed:
        st.divider()
        if st.button("2. Generate Source Overview"):
            with st.spinner(f"Writing Notebook Overview using {ai_choice}..."):
                try:
                    raw_text = st.session_state.raw_text
                    if ai_choice == "Gemini 2.5 Flash (Smart & Fast)":
                        llm_summary = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
                        prompt_text = f"You are an expert analyst. Read the following source material and provide a detailed, multi-paragraph overview of the key themes, main arguments, and essential data points. \n\n{raw_text[:50000]}"
                        response = llm_summary.invoke(prompt_text)
                        summary_content = response.content
                    elif ai_choice == "Claude 3.5 Sonnet (Nuanced & Logical)":
                        llm_summary = ChatAnthropic(model_name="claude-3-5-sonnet-latest", anthropic_api_key=anthropic_api_key)
                        prompt_text = f"You are an expert analyst. Read the following source material and provide a detailed, multi-paragraph overview of the key themes, main arguments, and essential data points. \n\n{raw_text[:50000]}"
                        response = llm_summary.invoke(prompt_text)
                        summary_content = response.content
                    else:
                        llm_summary = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=hf_api_token, temperature=0.3, max_new_tokens=512)
                        prompt_text = f"Provide a comprehensive summary of the key themes in this text:\n\n{raw_text[:8000]}"
                        summary_content = llm_summary.invoke(prompt_text)
                    
                    overview_message = f"**📑 Source Overview:**\n\n{summary_content}"
                    st.session_state.chat_history.append({"role": "assistant", "content": overview_message})
                    st.rerun()
                except Exception as summary_e:
                    st.warning(f"Could not generate automatic summary: {summary_e}")

# --- MAIN CHAT INTERFACE ---
st.divider()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Reverted chat input text to be cleaner
if user_question := st.chat_input("Ask a question about your sources..."):
    if st.session_state.vector_store is None:
        st.error("Please process your sources first!")
    elif ai_choice == "Mistral 7B (Free Open Source)" and not hf_api_token:
         st.error("Cannot use Mistral. Please add your Hugging Face API token to the environment variables.")
    elif ai_choice == "Claude 3.5 Sonnet (Nuanced & Logical)" and not anthropic_api_key:
         st.error("Cannot use Claude. Please add your Anthropic API key to the environment variables.")
    else:
        # Display the user's question
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Build the Memory Buffer
        formatted_chat_history = ""
        recent_history = [msg for msg in st.session_state.chat_history if not msg["content"].startswith("**📑 Source Overview:**")][-6:]
        for msg in recent_history:
            formatted_chat_history += f"{msg['role'].capitalize()}: {msg['content']}\n"

        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing deeply using {ai_choice}..."):
                
                if ai_choice == "Gemini 2.5 Flash (Smart & Fast)":
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
                elif ai_choice == "Claude 3.5 Sonnet (Nuanced & Logical)":
                    llm = ChatAnthropic(model_name="claude-3-5-sonnet-latest", anthropic_api_key=anthropic_api_key)
                else:
                    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=hf_api_token, temperature=0.3, max_new_tokens=512)
                
                # --- DYNAMIC KNOWLEDGE INSTRUCTION ---
                if knowledge_source == "Strictly Uploaded Sources":
                    knowledge_instruction = "IMPORTANT: Answer STRICTLY using the provided context. If the answer is not in the context, clearly state that the information is not present in the uploaded sources. Do not hallucinate or use outside knowledge."
                else:
                    knowledge_instruction = "IMPORTANT: Prioritize the provided context, but you are authorized to use your general outside knowledge to supplement the answer, provide broader context, or answer the question if the sources lack the information. If you use outside knowledge, explicitly state that you are doing so."

                system_prompt = (
                    "You are ESS Notebook, a highly analytical and expert AI research assistant for Environmental Standards Scotland. "
                    "Your primary goal is to provide deeply detailed, comprehensive, and exhaustive answers. "
                    "Always structure your answers logically using multiple paragraphs, and use bullet points to extract key lists, metrics, or regulations. "
                    f"{knowledge_instruction}\n\n"
                    "You have memory of the previous conversation. Use the 'Previous Conversation' block below to understand follow-up questions, code references, or refinements.\n\n"
                    "Previous Conversation:\n{chat_history}\n\n"
                    "Context: {context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                try:
                    response = rag_chain.invoke({
                        "input": user_question,
                        "chat_history": formatted_chat_history
                    })
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
