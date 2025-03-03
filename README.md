# CDP Chatbot

A FastAPI-based chatbot that provides information about Customer Data Platforms (CDPs) including Segment, mParticle, Lytics, and Zeotap.

## Technology Stack

### Backend
- **FastAPI**: A modern, high-performance web framework for building APIs with Python. Chosen for its asynchronous capabilities and automatic OpenAPI documentation.
- **Uvicorn**: ASGI server that serves the FastAPI application with high performance.
- **Jinja2**: Templating engine used to render the HTML interface.

### Frontend
- **HTML/CSS/JavaScript**: Simple frontend with a chat interface that asynchronously sends requests to the backend API.

### Natural Language Processing & Retrieval
- **LangChain**: Framework for developing applications powered by language models, used for document processing and retrieval.
- **HuggingFace Embeddings**: Used for converting text into vector embeddings with the `sentence-transformers/all-mpnet-base-v2` model.
- **FAISS**: Facebook AI Similarity Search library for efficient similarity search and clustering of dense vectors, enabling fast retrieval of relevant documents.

## Data Structures

### Vector Stores
The application uses FAISS vector stores to efficiently retrieve relevant information from CDP documentation. Each CDP has its own vector store:

```python
vector_stores = {
    "segment": FAISS index,
    "mparticle": FAISS index,
    "lytics": FAISS index,
    "zeotap": FAISS index
}
```

### Document Processing
1. **WebBaseLoader**: Loads documentation from specified URLs for each CDP.
2. **RecursiveCharacterTextSplitter**: Splits documents into manageable chunks with overlap to maintain context:
   - Chunk size: 1000 characters
   - Chunk overlap: 800 characters

### Query Processing Pipeline
1. **Relevance Check**: Simple keyword-based filtering to ensure questions are relevant.
2. **CDP Extraction**: Regular expressions to identify which CDP(s) the user is asking about.
3. **Single CDP Queries**: Retrieves information from the specific CDP's vector store.
4. **Cross-CDP Comparison**: Handles queries that mention multiple CDPs by retrieving and formatting information from each.

## Architecture

1. **Initialization**: On startup, the application loads web documentation for each CDP, splits it into chunks, and creates vector stores.
2. **API Endpoints**:
   - `GET /`: Serves the chat interface.
   - `POST /ask`: Processes questions and returns answers based on the relevant CDP documentation.
3. **Query Processing**: The application processes user questions by:
   - Checking relevance
   - Identifying mentioned CDPs
   - Retrieving relevant documentation chunks
   - Formatting the response

## Running the Application

The application runs on Replit and is configured to use port 3000:

```
uvicorn main:app --host 0.0.0.0 --port 3000
```

Make sure to install the required dependencies:

```
pip install fastapi langchain langchain-community uvicorn
```
