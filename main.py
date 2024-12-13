"""Question and Answer with RAG"""

import argparse
import os
import sys

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
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

    glossary_based_prompt = """
You must answer the question strictly based on the given context and glossary:
- **Answer in a structured format**. Use appropriate headings, bullet points, and indentation for clarity.
- If the term or abbreviation is **in the glossary**, use the **exact definition**. Do not alter, translate, or infer its meaning.
- If the term or abbreviation is **not in the glossary**, search for any related context or information in the provided context. If no relevant information is found:
  - **Only make an informed guess** or **infer** based on the **available context** if absolutely necessary. 
  - Do not speculate on terms or concepts beyond what the context allows. If no context or related information is available, respond with: "The term or abbreviation is not defined in the provided context or glossary, and no related information can be found."
- If the provided context is insufficient to answer the question, respond with: "The provided context does not contain sufficient information to answer this question."
- **All answers must be provided in Korean**, except for technical terms and abbreviations, which should remain in their **original form** and **not be translated**.
- **Do not output the entire Glossary** unless the question specifically requires it.
- **All responses must be in a structured Markdown format**:
  - Use headings (`#`, `##`, etc.) to organize sections.
  - Bullet points should be used for lists.
  - Use proper indentation to maintain hierarchy.
  - If necessary, **enclose code snippets** or commands in ```code block``` format.
- All responses must be made from a Tizen development perspective.

### Response Length:
- **Do not repeat answer**
- **Limit responses to a reasonable length**, ensuring the answer is clear but concise.

### Tool Usage and Installation Guide:
- You must answer the question strictly based on the given context section
- Provide a **detailed usage guide** for tools used in Tizen development, including installation steps, prerequisites (e.g., Tizen SDK, dependencies, environment setup), and usage examples.
- Offer **step-by-step instructions** for setting up the Tizen development environment, including configuration for any required infrastructure (e.g., devices, cloud platforms, CI/CD pipelines).

### Context:
{context}

### Question:
{question}

### Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=glossary_based_prompt,
    )

    model = ChatOllama(
        model=model_name,
        max_length=50,
        repetition_penalty=1.7,
        temperature=0,
        top_p=0.3,
        top_k=50,
        num_return_sequences=1,
        stop=["### Answer:"],
    )

    def format_docs(docs):
        # Including document source metadata in the response
        formatted_docs = "\n\n".join([d.page_content for d in docs])
        return (
            formatted_docs
            + "\n\n### Sources:\n"
            + "\n".join([d.metadata.get("source") for d in docs])
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
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
    llm_model_name = "gemma2:2b-instruct-q4_K_M"
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
