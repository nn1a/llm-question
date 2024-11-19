"""Question and Answer with RAG"""

import argparse
import os
import sys


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_ollama import ChatOllama




def load_and_split_markdown_files(folder_path, chunk_size=512, chunk_overlap=50):
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


def create_rag_chain(vectorstore: FAISS, model_name):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain.
    Args:
        vectorstore (FAISS): FAISS vectorstore object
        model_name (str): Name of the LLM to use for the chain
    Returns:
        Chain: RAG chain object
    """

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question - answering tasks. If possible, translate into Korean, but do not translate technical terms and use them in their original language. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Information that is not given should be used in a limited way. Let's think step-by-step.<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question} 
Context: {context} 
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOllama(
        model=model_name,
        max_length=50,
        repetition_penalty=1.2,
        temperature=0,
        top_p=0.3,
        top_k=50,
        num_return_sequences=1,
    )

    # model = HuggingFacePipeline.from_model_id(
    #     model_id="meta-llama/Llama-3.2-1B-Instruct",
    #     task="text-generation",
    #     device=0,
    #     pipeline_kwargs={"max_new_tokens": 2048, "repetition_penalty": 1.2, "temperature":0.1, "top_p":0.3, "top_k" : 50 },
    # )

    meta_index = {}

    for uid, doc in vectorstore.docstore._dict.items():
        source = doc.metadata['source']
        if source in meta_index:
            meta_index[source].append(uid)
        else:
            meta_index[source] = [uid]

    def format_docs(docs):
        # Including document source metadata in the response
        # formatted_docs = "\n\n".join([d.page_content for d in docs])
        sources = {d.metadata.get("source", "Unknown Source") for d in docs}
        print(sources)
        
        document_ids = set(d.metadata["source"] for d in docs)
        documents_uids = [meta_index[doc_id] for doc_id in document_ids]
        documents = []
        for doc_ids in documents_uids:
            for doc_id in doc_ids:
                documents.append(vectorstore.docstore._dict[doc_id])
        formatted_docs = "\n\n".join([d.page_content for d in documents])
        return formatted_docs + "\n\nSources:\n" + "\n".join(sources)

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
    llm_model_name = "llama3.2:1b"
    vectorstore_path = "vectorstore"

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
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
    for answer in rag_chain.stream(query):
        sys.stdout.write(answer)


# Entry point
if __name__ == "__main__":
    main()
