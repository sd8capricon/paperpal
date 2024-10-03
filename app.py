import os
import time
import streamlit as st
from utils import load_pdfs, clear_directory
from analyzers import AnalyzerRAG
from langchain_core.messages import HumanMessage, AIMessage

# Initialize a session state variable to track file processing status
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False
# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialize session state to model
if "model" not in st.session_state:
    st.session_state.model = AnalyzerRAG(model="gemini-1.5-flash", temperature=0)

# page config
st.set_page_config(page_title="PaperPal")

# App Title
st.title("PaperPal: Your friendly AI for navigating contracts. 📑")

# Sidebar Title
st.sidebar.title("Contract Files")

# File Uploader
contracts = st.sidebar.file_uploader(
    "Upload PDFs",
    accept_multiple_files=True,
    type=["pdf"],
)

main_plaeceholder = st.empty()

query = st.chat_input(
    "Ask question about your documents",
    disabled=False,
)


if len(contracts) == 0:
    st.session_state.files_processed = False
    st.session_state.messages = []
    clear_directory("uploads")

if len(contracts) > 0 and not st.session_state.files_processed:
    main_plaeceholder.text("Data Loading...Started...✅ ✅ ✅")
    time.sleep(2)
    file_upload_text = ""
    for contract in contracts:
        bytes_data = contract.read()
        file_upload_text = file_upload_text + f"Uploaded File:{contract.name}\n"
        main_plaeceholder.text(file_upload_text)
        time.sleep(0.5)
    # Load pdfs in readable format
    main_plaeceholder.text("Creating Chunks and Vector Store...")
    pdfs = load_pdfs(contracts, "uploads")
    # Load Analyzer
    st.session_state.model.vectorize_pdfs(pdfs)
    main_plaeceholder.text("Initializing Chains...")
    st.session_state.model.initialize_chains()
    time.sleep(1)
    main_plaeceholder.text("Files Processed...✅ ✅ ✅")
    time.sleep(1)
    # Run Risk Review Query
    main_plaeceholder.text("Running Risk Review ❗️")
    initial_query = """
        Detect potential risks in the contracts.
        For every detected risk, give a concise explanation.
    """
    answer, chat_history = st.session_state.model.invoke(query=initial_query)
    main_plaeceholder.text("Review Complete ✅")
    # Response Markdown
    md = f"Hi there 👋 this is the risk review of your documents  \n{answer}  \nYou can further assess the liabilities or specfic clauses within the documents"
    st.session_state.model.chat_history[1].content = md
    st.session_state.model.chat_history.pop(0)
    st.session_state.files_processed = True

if query:
    if st.session_state.files_processed:
        answer, chat_history = st.session_state.model.invoke(query)
    else:
        st.chat_message("assistant").write(
            "Unable to answer your queries until the documents are processed, please wait."
        )

# Display the chat messages
for message in st.session_state.model.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)
