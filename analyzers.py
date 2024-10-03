from dotenv import load_dotenv
from typing import Literal
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory


load_dotenv()


class AnalyzerRAG:
    max_tokens_per_min = 1e6
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", "."],
        chunk_size=1000,
        chunk_overlap=200,
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chat_history = []

    # Prompt Strings
    __contextualize_q_system_prompt = """
    Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """
    __qa_prompt_string = """
    You are a contract review assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that you don't know. Keep the answer concise..
    Never provide incorrect information. If it's a greeting respond nicely.

    Context: {context}
    """
    # Prompt Templates
    __contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", __contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    __qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", __qa_prompt_string),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def __init__(
        self,
        model: Literal["gemini-1.5-pro", "gemini-1.5-flash"],
        temperature=0,
        pdfs=None,
    ) -> None:

        if pdfs:
            self.chunks = self.__get_chunks(pdfs)
            self.vector_store = self.__get_vector_store(
                self.chunks,
                embeddings=self.embeddings,
            )
        self.llm = ChatGoogleGenerativeAI(temperature=temperature, model=model)

    def __get_chunks(self, pdfs):
        chunks = []
        for pdf in pdfs:
            pdf_chunk = self.text_splitter.split_documents(pdf)
            chunks.extend(pdf_chunk)
        return chunks

    def __get_vector_store(self, chunks, embeddings):
        return FAISS.from_documents(chunks, embedding=embeddings)

    def vectorize_pdfs(self, pdfs):
        self.chunks = self.__get_chunks(pdfs)
        self.vector_store = self.__get_vector_store(
            self.chunks, embeddings=self.embeddings
        )

    def initialize_chains(self):
        # Chains
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.vector_store.as_retriever(),
            self.__contextualize_q_prompt,
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, self.__qa_prompt)
        self.rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

    def invoke(self, query, chat_history=None):
        if chat_history:
            response = self.rag_chain.invoke(
                {"input": query, "chat_history": chat_history}
            )
        else:
            response = self.rag_chain.invoke(
                {"input": query, "chat_history": self.chat_history}
            )
        # update history
        self.chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=response["answer"]),
            ]
        )
        return response["answer"], self.chat_history


class AnalyzerQA:
    max_tokens_per_min = 1e6
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", "."],
        chunk_size=1000,
        chunk_overlap=200,
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def __init__(
        self,
        pdfs,
        model: Literal["gemini-1.5-pro", "gemini-1.5-flash"],
        temperature=0,
    ) -> None:
        self.chunks = self.__get_chunks(pdfs)
        self.vector_store = self.__get_vector_store(
            self.chunks,
            embeddings=self.embeddings,
        )
        self.llm = ChatGoogleGenerativeAI(temperature=temperature, model=model)
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
