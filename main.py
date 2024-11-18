"""Question and Answer with RAG"""

import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def load_and_split_markdown_files(folder_path, chunk_size=600, chunk_overlap=50):
    """
    Recursively loads and splits all Markdown files in a given folder.
    Args:
        folder_path (str): Path to the folder containing Markdown files
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between text chunks
    Returns:
        list: List of split LangChain Document objects
    """
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                print(f"Loading file: {file_path}")
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file_path

                documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_and_save_vectorstore(docs, embeddings, vectorstore_path="vectorstore"):
    """
    Creates and saves a FAISS vectorstore from documents using the given embeddings.
    Args:
        docs (list): List of split documents
        embeddings: Embedding model to use
        vectorstore_path (str): Directory to save the FAISS vectorstore
    Returns:
        FAISS: Created FAISS vectorstore object
    """
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vectorstore_path)
    print(f"Vectorstore saved to: {vectorstore_path}")
    return vectorstore


def load_vectorstore(vectorstore_path, embeddings):
    """
    Loads a saved FAISS vectorstore with the required embeddings.
    Args:
        vectorstore_path (str): Path to the saved FAISS vectorstore
        embeddings: Embedding model to use for loading the vectorstore
    Returns:
        FAISS: Loaded FAISS vectorstore object
    """
    print(f"Loading vectorstore from: {vectorstore_path}")
    return FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )


def create_rag_chain(vectorstore, model_name):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain.
    Args:
        vectorstore (FAISS): FAISS vectorstore object
        model_name (str): Name of the LLM to use for the chain
    Returns:
        Chain: RAG chain object
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template = """
You are an AI chatbot that provides detailed and polite answers based on the given context. You must answer in Korean.
and add Sources: information end of your answer

#Context
{context}

#Question
{question}

#Answer (Your answer should be in both English and Korean):
"""

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOllama(
        model=model_name,
        max_length=50,
        repetition_penalty=1.2,
        temperature=0.3,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1,
    )

    def format_docs(docs):
        # Including document source metadata in the response
        formatted_docs = "\n\n".join([d.page_content for d in docs])
        sources = "\n".join(
            [d.metadata.get("source", "Unknown Source") for d in docs]
        )  # Get the source file(s)
        return formatted_docs + "\n\nSources:\n" + sources

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


def main():
    """main entry point"""
    parser = argparse.ArgumentParser(
        description="Run a chatbot with optional embedding."
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        default=False,
        help="Perform embedding and create a new vectorstore",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=True,
        help="The question to ask the chatbot",
    )
    args = parser.parse_args()

    folder_path = "./tizen-docs"
    model_name = "intfloat/multilingual-e5-large-instruct"
    llm_model_name = "llama3.2"
    vectorstore_path = "vectorstore"

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Step 1: Embedding or load vectorstore
    if args.embedding or not os.path.exists(vectorstore_path):
        print("Embedding documents and creating vectorstore...")
        documents = load_and_split_markdown_files(folder_path)
        vectorstore = create_and_save_vectorstore(
            documents, embeddings, vectorstore_path=vectorstore_path
        )
    else:
        print("Loading existing vectorstore...")
        vectorstore = load_vectorstore(vectorstore_path, embeddings)

    # Step 2: Create the RAG chain
    print("Creating RAG chain...")
    rag_chain = create_rag_chain(vectorstore, llm_model_name)

    # Step 3: Query the RAG chain
    query = args.question  # Now using the question from the CLI
    print(f"Querying RAG chain with: {query}")
    answer = rag_chain.invoke(query)
    print(f"Answer:\n{answer}")


# Entry point
if __name__ == "__main__":
    main()
