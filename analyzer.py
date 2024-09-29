from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS

load_dotenv()


class Analyzer:
    max_tokens_per_min = 1e6
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", "."],
        chunk_size=1000,
        chunk_overlap=200,
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")

    def __init__(self, pdfs) -> None:
        self.chunks = self.__get_chunks(pdfs)
        self.vector_store = self.__get_vector_store(
            self.chunks,
            embeddings=self.embeddings,
        )
        self.chain = RetrievalQAWithSourcesChain.from_llm(
            llm=self.llm,
            max_tokens_limit=int(self.max_tokens_per_min),
            reduce_k_below_max_tokens=True,
            retriever=self.vector_store.as_retriever(),
            memory=ConversationBufferWindowMemory(
                k=4,
                memory_key="history",
                input_key="question",
                output_key="answer",
            ),
        )

    def __get_chunks(self, pdfs):
        chunks = []
        for pdf in pdfs:
            pdf_chunk = self.text_splitter.split_documents(pdf)
            chunks.extend(pdf_chunk)
        return chunks

    def __get_vector_store(self, chunks, embeddings):
        return FAISS.from_documents(chunks, embedding=embeddings)

    def invoke(self, query):
        response = self.chain({"question": query}, return_only_outputs=True)
        return response["answer"], response["sources"]
