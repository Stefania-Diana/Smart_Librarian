from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from generator import HyDEGenerator

class RAGSearch:
    def __init__(
        self,
        api_key: str,
        chroma_path: str,
        collection_name: str = "smart_librarian",
        embed_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4.1-mini",
        k: int = 4,
        use_hyde: bool = True
    ):
        self.api_key = api_key
        self.k = k
        self.use_hyde = use_hyde
        self.hyde = HyDEGenerator(api_key) if use_hyde else None

        self.client = OpenAI(api_key=api_key)
        self.llm_model = llm_model

        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=embed_model
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )

    def _retrieve(self, query: str) -> Dict:
        return self.collection.query(query_texts=[query], n_results=self.k)

    def _compose_prompt(self, question: str, context_blocks: List[str]) -> List[Dict[str, str]]:
        context = "\n\n".join([blk for blk in context_blocks])
        system = (
            "You are Smart Librarian. Answer using ONLY the provided context from classic book summaries. "
            "If the answer cannot be found, reply exactly with: "
            "'I don't know based on the available book summaries.' "
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Context (book summaries):\n{context}\n"
            "Instructions: Provide a concise answer first, then a short explanation with cited titles."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def ask(self, question: str) -> Dict:
        query = question
        hyde_text = None
        if self.use_hyde and self.hyde:
            hyde_text = self.hyde.generate(question)
            query = f"{question}\n\nHypothetical relevant answer:\n{hyde_text}"

        results = self._retrieve(query)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        messages = self._compose_prompt(question, docs)
        resp = self.client.chat.completions.create(
            model=self.llm_model, messages=messages, temperature=0.2, max_tokens=350
        )
        answer = resp.choices[0].message.content.strip()

        # Build human-friendly citations from metadatas
        citations = []
        for m in metas:
            title = m.get("title", "Unknown Title")
            pr = m.get("page_range", "?")

        return {
            "hyde": hyde_text,
            "answer": answer
        }
