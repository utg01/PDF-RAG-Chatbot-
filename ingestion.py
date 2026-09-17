from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
import re

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

@tool
def Create_embeddings(file_path: str, file_name: str) -> dict:
    """
    Use this tool when the user uploads a PDF file.

    This tool:
    - Loads the PDF using PyPDFLoader
    - Splits it into chunks and generates vector embeddings
    - Stores the embeddings in a Pinecone vector database

    Args:
        file_path: The local path where the uploaded PDF is temporarily saved
        file_name: The original name of the uploaded PDF file

    Returns on success:
    {
        "status": "success",
        "file_name": <name>,
        "index_name": <pinecone index name>
    }
    Returns on failure:
        Error message string
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            return "Could not extract any text from the PDF"

        # splitting
        all_splits = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(docs)

        # index name
        index_name = f"pdf-{file_name}"
        index_name = re.sub(
            r'[^a-z0-9-]',
            '-',
            index_name.lower()
        )
        index_name = re.sub(
            r'-+',
            '-',
            index_name
        ).strip('-')
        index_name = index_name[:45]

        # getting embedding dimensions
        embedding_dimension = len(
            embeddings.embed_query("hello")
        )

        # creating index
        existing_indexes = pc.list_indexes().names()

        if index_name in existing_indexes:
            existing_dimension = pc.describe_index(
                index_name
            ).dimension

            # deleting wrong-dimension index automatically
            if existing_dimension != embedding_dimension:
                pc.delete_index(index_name)

        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        # storing embeddings
        PineconeVectorStore.from_documents(
            documents=all_splits,
            embedding=embeddings,
            index_name=index_name
        )

        return {
            "status": "success",
            "file_name": file_name,
            "index_name": index_name
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Create_embeddings Error: {str(e)}"

@tool
def delete_index(index_name: str) -> str:
    """
    Deletes a Pinecone index. Call this when the user starts a new session
    or explicitly ends the current one, to clean up stored embeddings.

    Args:
        index_name: The Pinecone index name to delete

    Returns:
        A status message
    """
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            pc.delete_index(index_name)
            return f"Index {index_name} deleted successfully"
        return f"Index {index_name} does not exist"
    except Exception as e:
        return f"delete_index Error: {str(e)}"