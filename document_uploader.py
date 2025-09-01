import os
import re
from typing import List, Dict, Tuple
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

class DocumentUploader:
    def __init__(
        self,
        api_key: str,
        data_dir: str,
        chroma_path: str,
        collection_name: str = "smart_librarian",
        embed_model: str = "text-embedding-3-small",
        chunk_chars: int = 900,
        overlap: int = 200
    ):
        self.api_key = api_key
        self.data_dir = data_dir
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.chunk_chars = chunk_chars
        self.overlap = overlap

        os.makedirs(self.chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name=embed_model
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    # PDF parsing tailored to the provided database
    def _load_pdf_text(self, path: str) -> Tuple[str, List[Tuple[int, str]]]:
        reader = PdfReader(path)
        page_texts = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            page_texts.append((i + 1, txt))
        # Return full concatenation for simple chunking,
        # alongside per-page list to estimate page ranges for chunks.
        full_text = "\n".join(t for _, t in page_texts)
        return full_text, page_texts

    def _split_by_titles(self, text: str) -> List[Dict]:
        """
        Your PDF uses 'Title: <Book Name>' followed by a summary and 'Themes:'.
        We split on 'Title:' and keep the title + following paragraph(s).
        """
        parts = re.split(r"\bTitle:\s*", text)
        docs = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # First line until newline is the title
            first_line, _, rest = part.partition("\n")
            title = first_line.strip()
            body = rest.strip()
            if not title:
                continue
            docs.append({"title": title, "body": body})
        return docs

    def _chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(end - self.overlap, end)
        return chunks

    def _guess_page_span(self, chunk: str, page_texts: List[Tuple[int, str]]) -> str:
        """
        Heuristic: find pages whose text overlaps this chunk.
        Returns a small page span string like '1-2' or '3'.
        """
        hit_pages = []
        for pno, ptxt in page_texts:
            # very small heuristic match
            if any(seg in ptxt for seg in [chunk[:80], chunk[-80:]]):
                hit_pages.append(pno)
        if not hit_pages:
            return "?"
        return f"{min(hit_pages)}-{max(hit_pages)}" if len(hit_pages) > 1 else f"{hit_pages[0]}"

    def _upsert_docs(self, source_pdf: str, docs: List[Dict], page_texts: List[Tuple[int, str]]):
        ids, metadatas, documents = [], [], []
        for i, d in enumerate(docs):
            title = d["title"]
            for j, chunk in enumerate(self._chunk(d["body"])):
                ids.append(f"{title}-{i}-{j}")
                metadatas.append({
                    "title": title,
                    "source": os.path.basename(source_pdf),
                    "page_range": self._guess_page_span(chunk, page_texts)
                })
                documents.append(chunk)

        if not documents:
            return

        # Avoid duplicate ids on re-run: Chroma will throw if ids exist.
        # Strategy: try add; if conflict, delete & re-add.
        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception:
            self.collection.delete(ids=ids)
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def ingest(self, reset: bool = False):
        if reset:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

        # Index all PDFs in /data
        for fname in os.listdir(self.data_dir):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(self.data_dir, fname)
                full_text, page_texts = self._load_pdf_text(path)
                docs = self._split_by_titles(full_text)
                self._upsert_docs(source_pdf=path, docs=docs, page_texts=page_texts)
