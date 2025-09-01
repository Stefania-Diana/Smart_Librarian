# Smart Librarian

**Smart Librarian** is a Streamlit-based AI assistant that answers questions about books using a Retrieval-Augmented Generation (RAG) pipeline. It uses OpenAI for embeddings and chat completions, performs HyDE-style hypothetical reasoning, filters inappropriate inputs, and supports optional text-to-speech output.


## Features

- **PDF Ingestion**: Upload book summary PDFs into a Chroma vector store.
- **RAG-Based QA**: Retrieve relevant chunks from indexed documents and answer using GPT.
- **HyDE (Hypothetical Answer Generation)**: Improve retrieval through speculative prompts.
- **Safety Filtering**: Detect profanity and OpenAI moderation flags.
- **Text-to-Speech (TTS)**: Read answers aloud.


## Directory Structure

```
project-root/
├── app.py                 # Entry point to start ingestion + Streamlit app
├── document_uploader.py   # PDF parsing, chunking, embedding, and ingestion logic
├── search_RAG.py          # RAG logic + HyDE integration + answer generation
├── generator.py           # HyDE hypothetical generation logic
├── filters.py             # SafetyFilter with profanity and moderation API
├── tts.py                 # Local TTS (text-to-speech)
├── ui.py                  # Streamlit interface
├── requirements.txt       # Python dependencies
└── data/                  # Folder for PDFs and generated audio
```

### Step 1: Install dependencies
pip install -r requirements.txt
```

##Add your PDFs
Place your book summary PDFs in the `data/` folder.
```

### Step 2: Launch the app
```bash
python -m streamlit run app.py
```
This will:
- Ingest and embed your PDFs into a local ChromaDB vector store.
- Start the Streamlit web app at `http://localhost:8501/`


## Optional Enhancements
- Enable HyDE using the checkbox in the UI to generate a better search query.
- Enable "Read aloud" for TTS-based answer playback.


## Requirements
Dependencies listed in `requirements.txt`:
```text
python-dotenv
openai>=1.30.0
chromadb>=0.5.0
pypdf
streamlit
pyttsx3
```


