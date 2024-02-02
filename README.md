# Medical Analyzer Bot with Retrieval-based QA

## Overview

This repository contains the implementation of a Medical Bot designed to answer user queries based on pre-existing medical documents. The bot uses a retrieval-based question-answering (QA) approach, incorporating language models and vector stores for efficient information retrieval.

## Tech Stack

- **Language Model**: [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) - A large language model for conversational AI.
- **Embeddings Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Hugging Face embeddings model for transforming sentences into vectors.
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors.
- **Document Loaders**: [PyPDFLoader](langchain_community/document_loaders.py), [DirectoryLoader](langchain_community/document_loaders.py) - Loaders for extracting text from PDFs and directories of documents.
- **Text Splitter**: [RecursiveCharacterTextSplitter](langchain/text_splitter.py) - Splits documents into chunks for efficient processing.

## Project Structure

- `model.py`: Python script defining the QA model, retrieval chain, and functions to load and utilize the models.
- `ingest.py`: Python script for ingesting medical documents, creating vector representations, and saving them using FAISS.
- `data/`: Directory containing medical documents in PDF format.
- `vectorstore/`: Directory to store the FAISS vector database (`db_faiss`).

## Usage

### Setting Up the Vector Database

1. Ensure the required Python dependencies are installed: `pip install -r requirements.txt`.
2. Run `ingest.py` to ingest medical documents and create the FAISS vector database.

### Running the Medical Bot

1. Install the necessary dependencies: `pip install -r requirements.txt`.
2. Run `model.py` to start the Medical Bot.
3. The bot will prompt users for queries, and it will provide answers based on the pre-existing medical documents.

## Contributions

Contributions are welcome! Feel free to open issues, suggest improvements, or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
