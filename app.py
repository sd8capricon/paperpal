import os
import time
import streamlit as st
from utils import load_pdfs, clear_directory
from analyzers import AnalyzerRAG

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
st.title("PaperPal: Your friendly AI for navigating documents and contracts. 📑")

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
        For each contract
        Detect potential risks.
        For each risk give its source document name.
        """
    answer, chat_history = st.session_state.model.invoke(query=initial_query)
    main_plaeceholder.text("Review Complete ✅")
    # Response Markdown
    md = f"Hi there 👋 this is the risk review of your documents  \n{answer}  \nYou can further assess the liabilities or specfic clauses within the documents"
    # st.chat_message("assistant").markdown(md)
    st.session_state.messages.append({"role": "assistant", "content": md})
    st.session_state.files_processed = True

if query:
    if st.session_state.files_processed:
        # Append user query to the chat history
        st.session_state.messages.append({"role": "human", "content": query})

        # Append assistant response
        answer, sources = st.session_state.model.invoke(query)
        md = f"{answer}  \nSources: {sources}"
        st.session_state.messages.append({"role": "assistant", "content": md})
    else:
        st.chat_message("assistant").write(
            "Unable to answer your queries until the documents are processed, please wait."
        )

# Display the chat messages
for message in st.session_state.messages:
    if message["role"] == "human":
        st.chat_message("human").markdown(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").markdown(message["content"])
