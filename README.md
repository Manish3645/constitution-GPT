# RAG-Based Streamlit Application

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, Hugging Face embeddings, ChromaDB, and Ollama (Phi-3).

The chatbot answers questions based only on the provided document instead of relying solely on the language model's knowledge. This helps reduce hallucinations and improves the accuracy of responses.

---

## What document did you use and why?

I used a PDF containing study notes related to the chosen topic.

For example, in this implementation I used a document containing educational content and notes relevant to the selected domain. The document was chosen because it provides structured information that can be retrieved and used to answer user queries accurately.

Using a domain-specific document ensures that the chatbot remains focused on the selected topic and reduces the chances of generating unrelated responses.

---

## How does your chunking work?

The document is split using LangChain's RecursiveCharacterTextSplitter.

Configuration:

* Chunk Size: 500 characters
* Chunk Overlap: 50 characters

This approach:

* Breaks large documents into smaller manageable sections.
* Preserves context between chunks through overlap.
* Improves retrieval accuracy by ensuring relevant information is available during search.

---

## Which embedding model did you use?

Embedding Model:

sentence-transformers/all-MiniLM-L6-v2

Reason for selection:

* Lightweight and efficient.
* Produces high-quality semantic embeddings.
* Widely used in RAG applications.
* Works well on local systems without requiring high-end hardware.

The embedding model converts document chunks into vector representations which are stored in ChromaDB.

---

## Technologies Used

* Streamlit
* LangChain
* Hugging Face Embeddings
* ChromaDB
* Ollama
* Phi-3
* PyPDF
* Python

---

## Architecture

Document PDF
↓
Document Loading
↓
Text Chunking
↓
Embedding Generation
↓
ChromaDB Vector Store
↓
Retriever
↓
Phi-3 (Ollama)
↓
Answer Generation

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Ollama

```bash
ollama run phi3
```

### 6. Run Streamlit application

```bash
streamlit run app.py
```

### 7. Open Browser

```text
http://localhost:8501
```

---

## Screenshot

Add a screenshot of:

* Running Streamlit application
* User query
* Generated response

Example:

screenshots/app-demo.png

---

## What would you improve with more time?

If given more time, I would:

1. Support multiple PDF uploads.
2. Add conversation memory.
3. Display source citations for retrieved answers.
4. Improve UI using Streamlit components.
5. Add document upload through the web interface.
6. Optimize retrieval using hybrid search techniques.
7. Deploy the application on Hugging Face Spaces or a cloud platform.
8. Add authentication and user management.

---

## Conclusion

This project demonstrates a basic RAG pipeline using Streamlit, Hugging Face embeddings, ChromaDB, and Ollama. By retrieving relevant information from the document before generating responses, the chatbot provides more accurate and context-aware answers while reducing hallucinations.
