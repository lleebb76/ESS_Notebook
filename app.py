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
from langchain_openai import ChatOpenAI
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

st.markdown("### Your research and coding assistant for Environmental Standards.")

# --- NEW INTRODUCTION ---
st.info("""
**Welcome to ESS Notebook! Here is how to use this tool:**
1. **Add Your Sources:** You do not have to add sources - if you select "Include General Knowledge" in the Knowledge Base section on the left of the screen, you can use this app just like a normal AI chat. If you want to use it as a Notebook, you can upload sources. If you select "Strictly Uploaded Sources" in the Knowledge Base section on the left, the app will only look at the sources you provided, nothing else from the wider internet, but if you select "Include General Knowledge" the app will combine your sources with others on the web. Upload your PDFs, Word docs, spreadsheets, or paste web links in the sidebar, then click **Process & Index Sources**.
2. **Generate an Overview:** Once processed, you can click **Generate Source Overview** to get an automatic, detailed summary of all your materials.
3. **Chat & Analyze:** Ask deep, complex questions about your sources in the chat box below.
* **Tip:** Use the settings in the sidebar to change the AI's "Brain," or switch the Knowledge Base to **"Include General Knowledge"** to use the AI as a standard chatbot without needing to upload any sources!
""")

# --- API KEY CHECKS ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")
hf_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not gemini_api_key:
    st.warning("⚠️ Please set your GEMINI_API_KEY in the environment variables.")
if not hf_api_token:
    st.error("🛑 HUGGINGFACEHUB_API_TOKEN is missing. This is required to process documents for free.")
if not anthropic_api_key:
    st.info("ℹ️ ANTHROPIC_API_KEY not found. The Claude option will be disabled.")
if not openai_api_key:
    st.info("ℹ️ OPENAI_API_KEY not found. The ChatGPT option will be disabled.")

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
    
    # Updated AI Choice with Llama 3
    ai_choice = st.radio(
        "Select your AI Brain:",
        (
            "Gemini 2.5 Flash (Smart & Fast)", 
            "Gemini 2.5 Pro (Advanced Reasoning)", 
            "ChatGPT (OpenAI)", 
            "Claude 3.5 Sonnet (Nuanced & Logical)", 
            "Llama 3 8B (Free Open Source)",
            "Mistral 7B (Free Open Source)"
        )
    )
    
    # Knowledge Base Toggle
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
                    
                    # Logic updated to include new models
                    if ai_choice == "Gemini 2.5 Flash (Smart & Fast)":
                        llm_summary = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
                    elif ai_choice == "Gemini 2.5 Pro (Advanced Reasoning)":
                        llm_summary = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=gemini_api_key)
                    elif ai_choice == "ChatGPT (OpenAI)":
                        llm_summary = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                    elif ai_choice == "Claude 3.5 Sonnet (Nuanced & Logical)":
                        llm_summary = ChatAnthropic(model_name="claude-3-5-sonnet-latest", anthropic_api_key=anthropic_api_key)
                    elif ai_choice == "Llama 3 8B (Free Open Source)":
                        llm_summary = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=hf_api_token, temperature=0.3, max_new_tokens=512)
                    else:
                        llm_summary = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=hf_api_token, temperature=0.3, max_new_tokens=512)
                    
                    # Process the prompt
                    if "Mistral" in ai_choice or "Llama" in ai_choice:
                        prompt_text = f"Provide a comprehensive summary of the key themes in this text:\n\n{raw_text[:8000]}"
                    else:
                        prompt_text = f"You are an expert analyst. Read the following source material and provide a detailed, multi-paragraph overview of the key themes, main arguments, and essential data points. \n\n{raw_text[:50000]}"
                    
                    response = llm_summary.invoke(prompt_text)
                    summary_content = response.content if hasattr(response, 'content') else response
                    
                    overview_message = f"**📑 Source Overview:**\n\n{summary_content}"
                    st.session_state.chat_history.append({"role": "assistant", "content": overview_message})
                    st.rerun()
                except Exception as summary_e:
                    error_msg = str(summary_e)
                    if "limit: 0" in error_msg and "gemini-2.5-pro" in error_msg:
                        st.error("🛑 Billing Error: Your Google API key is on the Free Tier, which does not grant access to Gemini 2.5 Pro. Please update your billing in Google AI Studio or use Gemini Flash.")
                    elif "credit balance is too low" in error_msg:
                        st.error("🛑 Billing Error: Your Anthropic API account has run out of credits. Please add prepaid funds in the Anthropic Console to use Claude.")
                    else:
                        st.warning(f"Could not generate automatic summary: {summary_e}. If using Llama 3, ensure you have accepted the license on Hugging Face.")

# --- MAIN CHAT INTERFACE ---
st.divider()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question about your sources (or use general knowledge)..."):
    
    # --- NO-SOURCE BYPASS LOGIC ---
    if st.session_state.vector_store is None and knowledge_source == "Strictly Uploaded Sources":
        st.error("Please process your sources first, or switch the Knowledge Base to 'Include General Knowledge' to chat freely!")
    elif ai_choice == "Mistral 7B (Free Open Source)" and not hf_api_token:
         st.error("Cannot use Mistral. Please add your Hugging Face API token to the environment variables.")
    elif ai_choice == "Llama 3 8B (Free Open Source)" and not hf_api_token:
         st.error("Cannot use Llama 3. Please add your Hugging Face API token to the environment variables.")
    elif ai_choice == "Claude 3.5 Sonnet (Nuanced & Logical)" and not anthropic_api_key:
         st.error("Cannot use Claude. Please add your Anthropic API key to the environment variables.")
    elif ai_choice == "ChatGPT (OpenAI)" and not openai_api_key:
         st.error("Cannot use ChatGPT. Please add your OpenAI API key to the environment variables.")
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
                
                # Model Initialization
                if ai_choice == "Gemini 2.5 Flash (Smart & Fast)":
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)
                elif ai_choice == "Gemini 2.5 Pro (Advanced Reasoning)":
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=gemini_api_key)
                elif ai_choice == "ChatGPT (OpenAI)":
                    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                elif ai_choice == "Claude 3.5 Sonnet (Nuanced & Logical)":
                    llm = ChatAnthropic(model_name="claude-3-5-sonnet-latest", anthropic_api_key=anthropic_api_key)
                elif ai_choice == "Llama 3 8B (Free Open Source)":
                    llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=hf_api_token, temperature=0.3, max_new_tokens=512)
                else:
                    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=hf_api_token, temperature=0.3, max_new_tokens=512)
                
                # --- DYNAMIC KNOWLEDGE INSTRUCTION ---
                if knowledge_source == "Strictly Uploaded Sources":
                    knowledge_instruction = "IMPORTANT: Answer STRICTLY using the provided context. If the answer is not in the context, clearly state that the information is not present in the uploaded sources. Do not hallucinate or use outside knowledge."
                else:
                    knowledge_instruction = "IMPORTANT: Prioritize the provided context if it exists, but you are completely authorized to use your general outside knowledge to answer the question or provide broader context."

                system_prompt = (
                    "You are ESS Notebook, a highly analytical and expert AI research assistant for Environmental Standards Scotland. "
                    "Your primary goal is to provide deeply detailed, comprehensive, and exhaustive answers. "
                    "Always structure your answers logically using multiple paragraphs, and use bullet points to extract key lists, metrics, or regulations. "
                    f"{knowledge_instruction}\n\n"
                    "You have memory of the previous conversation. Use the 'Previous Conversation' block below to understand follow-up questions, code references, or refinements.\n\n"
                    "Previous Conversation:\n{chat_history}\n\n"
                    "Context: {context}"
                )
                
                try:
                    # RAG vs Non-RAG Logic
                    if st.session_state.vector_store is not None:
                        # RAG Route (Sources uploaded)
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
                        question_answer_chain = create_stuff_documents_chain(llm, prompt)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        response = rag_chain.invoke({
                            "input": user_question,
                            "chat_history": formatted_chat_history
                        })
                        answer = response["answer"]
                    else:
                        # General Knowledge Route (No sources uploaded)
                        system_prompt_no_context = system_prompt.replace("Context: {context}", "Context: No documents uploaded.")
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt_no_context),
                            ("human", "{input}"),
                        ])
                        chain = prompt | llm
                        response = chain.invoke({
                            "input": user_question,
                            "chat_history": formatted_chat_history
                        })
                        answer = response.content if hasattr(response, 'content') else response

                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = str(e)
                    if "limit: 0" in error_msg and "gemini-2.5-pro" in error_msg:
                        st.error("🛑 Billing Error: Your Google API key is on the Free Tier, which does not grant access to Gemini 2.5 Pro. Please update your billing in Google AI Studio or use Gemini Flash.")
                    elif "credit balance is too low" in error_msg:
                        st.error("🛑 Billing Error: Your Anthropic API account has run out of credits. Please add prepaid funds in the Anthropic Console to use Claude.")
                    elif "403 Client Error" in error_msg and "Llama" in ai_choice:
                        st.error("🛑 Access Error: You must accept Meta's license agreement on Hugging Face before using Llama 3. See instructions below.")
                    else:
                        st.error(f"An error occurred while generating the response: {e}")
