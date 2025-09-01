Smart Librarian

Smart Librarian is a Streamlit-based AI assistant that answers questions about books using a Retrieval-Augmented Generation (RAG) pipeline. It uses OpenAI for embeddings and chat completions, performs HyDE-style hypothetical reasoning, filters inappropriate inputs, and supports optional text-to-speech output.

Features
Upload book summary PDFs into a Chroma vector store.
Retrieve relevant chunks from indexed documents and answer using GPT.
Improve retrieval through speculative prompts.
Detect profanity and OpenAI moderation flags.
Read answers aloud.

Structure
project-root/
├── app.py # Entry point to start ingestion + Streamlit app
├── document_uploader.py # PDF parsing, chunking, embedding, and ingestion logic
├── search_RAG.py # RAG logic + HyDE integration + answer generation
├── generator.py # HyDE hypothetical generation logic
├── filters.py # SafetyFilter with profanity and moderation API
├── tts.py # Local TTS (text-to-speech)
├── ui.py # Streamlit interface
├── requirements.txt # Python dependencies
└── data/ # Folder for PDFs and generated audio

Installation

Dependencies
Smart Librarian requires:
python-dotenv
openai>=1.30.0
chromadb>=0.5.0
pypdf
streamlit
pyttsx3

Install dependencies
pip install -r requirements.txt

How to Run
Step 1: Add your PDFs
Place your book summary PDFs in the data/ folder.

Step 2: Launch the app
python -m streamlit run app.py

Optional Enhancements
Set reset=True in app.py to re-ingest PDFs from scratch.
Enable HyDE using the checkbox in the UI to generate a better search query.
Enable "Read aloud" for TTS-based answer playback.

License

MIT License (or specify your license).