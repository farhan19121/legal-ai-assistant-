"""Build the local legal vector database. Run once:  python ingest.py

Put PDFs / .txt files of statutes and judgments inside the data/ folder first.
"""
import glob
import os
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from legal_tools import DB_DIR, get_embeddings

load_dotenv()

DATA_DIR = "data"
BATCH = 40  # small batches + pause to stay under Gemini embedding rate limits


def load_documents():
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True):
        try:
            pages = PyPDFLoader(path).load()
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        for p in pages:
            p.metadata["source_name"] = os.path.splitext(os.path.basename(path))[0]
            p.metadata["page"] = int(p.metadata.get("page", 0)) + 1
        docs.extend(pages)
    for path in glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True):
        if os.path.basename(path).lower().startswith("readme"):
            continue
        loaded = TextLoader(path, encoding="utf-8").load()
        for p in loaded:
            p.metadata["source_name"] = os.path.splitext(os.path.basename(path))[0]
            p.metadata["page"] = 1
        docs.extend(loaded)
    return docs


def main():
    docs = load_documents()
    if not docs:
        print(f"No PDFs/txt files found in ./{DATA_DIR}. Add some and run again.")
        return
    chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    store = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    for i in range(0, len(chunks), BATCH):
        store.add_documents(chunks[i : i + BATCH])
        print(f"  embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")
        time.sleep(1)
    print(f"Done. Database saved in ./{DB_DIR}")


if __name__ == "__main__":
    main()
