# Medical Analyzer Bot with Retrieval-based QA

## Overview

This repository contains the implementation of a Medical Bot designed to answer user queries based on input pdf of the user's medical report. The bot uses a retrieval-based question-answering (QA) approach, incorporating language models and vector stores for efficient information retrieval.
The idea involves developing an end-to-end application for medical report analysis using state-of-the-art Natural Language Processing (NLP) technology. The application aims to streamline the interpretation of medical reports, ensuring faster and more accurate diagnoses. Additionally, a conversational agent will be implemented to answer queries related to the analysis.

## Team Members:
1. **Keerthana G** --> Shiv Nadar University, Chennai --> kethykrish23@gmail.com
2. **Karthick NG** --> Shiv Nadar University, Chennai --> nagakarthick2004@gmail.com 
   
## Tech Stack

- **Language Model**: [TheBloke/Llama-2-7B-Chat-](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin) - A large language model for conversational AI.
- **Embeddings Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Hugging Face embeddings model for transforming sentences into vectors.
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors.
- **Document Loaders**: [PyPDFLoader](langchain_community/document_loaders.py), [DirectoryLoader](langchain_community/document_loaders.py) - Loaders for extracting text from PDFs and directories of documents.
- **Text Splitter**: [RecursiveCharacterTextSplitter](langchain/text_splitter.py) - Splits documents into chunks for efficient processing.
- **Chains**
## Project Structure

- `model.py`: Python script defining the QA model, retrieval chain, and functions to load and utilize the models.
- `ingest.py`: Python script for ingesting medical documents, creating vector representations, and saving them using FAISS.
- `data/`: Directory containing medical report in PDF format.
- `vectorstore/`: Directory to store the FAISS vector database (`db_faiss`).


## Architecture
![Comsys](https://github.com/KeerthanaG23/Medical-Report-Analyser/assets/116378322/4538db50-8653-4466-a869-9142debc0fb3)

## Usage

## Installation

### Clone this repository to your local machine
```bash
git clone https://github.com/your-username/langchain-medical-bot.git
cd langchain-medical-bot
```

### (Optional but recommended) Create a Python virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### Install the required Python packages
```bash
pip install -r requirements.txt
```

### Run ingest to load data
```bash
python ingest.py
```

### Run chainlit
```bash
chainlit run model.py -w
```

### Setting Up the Vector Database

1. Ensure the required Python dependencies are installed: `pip install -r requirements.txt`.
2. Run `ingest.py` to ingest medical documents and create the FAISS vector database. (python ingest.py)

### Running the Medical Bot

1. Run `model.py` to start the Medical Bot.
2. The bot will prompt users for queries, and it will provide answers based on the pre-existing medical documents.


