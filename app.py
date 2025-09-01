import os
from dotenv import load_dotenv

from document_uploader import DocumentUploader
from ui import StreamlitUI

DATA_DIR = "data"
CHROMA_PATH = os.path.join(DATA_DIR, "chroma")

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in environment (.env).")

    # Build / update the local vector store from PDFs in /data
    ingestor = DocumentUploader(
        api_key=api_key,
        data_dir=DATA_DIR,
        chroma_path=CHROMA_PATH,
        collection_name="smart_librarian"
    )
    # set reset=True if you want to rebuild from scratch
    ingestor.ingest(reset=False)

    # Launch Streamlit UI (RAG + HyDE + safety + TTS)
    ui = StreamlitUI(
        api_key=api_key,
        chroma_path=CHROMA_PATH,
        collection_name="smart_librarian"
    )
    ui.launch()

if __name__ == "__main__":
    main()
