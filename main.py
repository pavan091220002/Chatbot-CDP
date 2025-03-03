import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import uvicorn

os.environ["USER_AGENT"] = "MyCDPChatbot/1.0"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

CDP_URLS = {
    "segment": ["https://segment.com/docs/connections/sources/"],
    "mparticle": ["https://docs.mparticle.com/guides/getting-started/"],
    "lytics": ["https://docs.lytics.com/docs"],
    "zeotap": ["https://docs.zeotap.com/home/en-us/"]
}

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CDP_NAMES = ["segment", "mparticle", "lytics", "zeotap"]

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def initialize_vector_stores():
    vector_stores = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=800
    )

    for cdp, urls in CDP_URLS.items():
        loader = WebBaseLoader(urls)
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_stores[cdp] = vector_store

    return vector_stores

vector_stores = initialize_vector_stores()

def is_relevant(question):
    """Basic relevance check using keywords"""
    keywords = ["how to", "how do i", "create", "set up", "configure", "integrate"]
    return any(kw in question.lower() for kw in keywords)

def extract_cdps(question):
    """Extract CDP names from question using regex"""
    pattern = re.compile(r'\b(' + '|'.join(CDP_NAMES) + r')\b', re.IGNORECASE)
    return list(set([match.lower() for match in pattern.findall(question)]))

def handle_query(question):
    """Main processing pipeline"""
    if not is_relevant(question):
        return "I specialize in answering how-to questions about CDPs. Please ask a question related to Segment, mParticle, Lytics, or Zeotap."

    cdps = extract_cdps(question)

    if not cdps:
        return "Please specify which CDP you're asking about (Segment, mParticle, Lytics, or Zeotap)."

    if len(cdps) > 1:
        return compare_cdps(cdps, question)

    return answer_single_cdp(cdps[0], question)

def answer_single_cdp(cdp, question):
    """Retrieve answer for single CDP"""
    docs = vector_stores[cdp].similarity_search(question, k=3)
    return format_answer(docs)

def compare_cdps(cdps, question):
    """Handle cross-CDP comparison (bonus feature)"""
    comparisons = []
    for cdp in cdps:
        docs = vector_stores[cdp].similarity_search(question, k=2)
        comparisons.append(f"{cdp.capitalize()}:\n{format_answer(docs)}\n")
    return "\n".join(comparisons)

def format_answer(docs):
    """Format retrieved documents into readable answer"""
    return "\n\n".join([f"From {doc.metadata['source']}:\n{doc.page_content}"
                      for doc in docs])

@app.post('/ask')
async def ask(request: Request):
    question = (await request.json()).get('question', '')
    response = handle_query(question)
    return JSONResponse({"answer": response})

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def cli():
    print("CDP Support Chatbot - Type your question or 'exit' to quit")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break
        print("\nAnswer:", handle_query(question))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
