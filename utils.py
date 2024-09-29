import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", "\n\n", "."],
    chunk_size=1000,
    chunk_overlap=200,
)


def load_pdfs(contracts, directory):
    pdfs = []
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    for contract in contracts:
        path = os.path.join("uploads", contract.name)
        with open(path, "wb") as f:
            f.write(contract.getvalue())
        pdf_loader = PyPDFLoader(path)
        pdf = pdf_loader.load()
        pdfs.append(pdf)
    return pdfs


def clear_directory(directory):
    for filename in os.listdir(directory):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)

            # Remove the PDF file
            os.remove(file_path)
            print(f"Removed: {file_path}")
