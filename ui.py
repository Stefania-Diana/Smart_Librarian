import streamlit as st
from search_RAG import RAGSearch
from filters import SafetyFilter
from tts import synthesize_to_wav

class StreamlitUI:
    def __init__(self, api_key: str, chroma_path: str, collection_name: str):
        self.api_key = api_key
        self.chroma_path = chroma_path
        self.collection_name = collection_name

    def launch(self):
        st.set_page_config(page_title="Smart Librarian", layout="centered")
        st.markdown(
            """
            <style>
                body {
                    background-color: #FF5640;
                }
                [data-testid="stAppViewContainer"] {
                    background-color: #192B37
                }
            </style>
            """,
        unsafe_allow_html=True
        )
        st.title("Smart Librarian")
        st.markdown(
            "Ask questions about books: "
        )

        # Controls
        use_hyde = st.checkbox("Enable HyDE ", value=False)
        read_aloud = st.checkbox("Read the answer aloud", value=False)

        question = st.text_input("Your question about the books:", placeholder="e.g., What is the main theme of The Great Gatsby?")
        ask = st.button("Ask")

        if ask:
            if not question.strip():
                st.error("Please type a question.")
                return

            # Safety first
            guard = SafetyFilter(api_key=self.api_key)
            flagged, reasons = guard.check(question)
            if flagged:
                st.error("Your input appears to contain inappropriate language and was blocked.")
                with st.expander("Why was it blocked?"):
                    for r in reasons:
                        st.write(f"- {r}")
                return

            with st.spinner("Thinking..."):
                rag = RAGSearch(
                    api_key=self.api_key,
                    chroma_path=self.chroma_path,
                    collection_name=self.collection_name,
                    use_hyde=use_hyde,
                )
                result = rag.ask(question)

            if use_hyde and result.get("hyde"):
                st.subheader("HyDE (hypothetical pre-answer)")
                st.code(result["hyde"], language="markdown")

            st.subheader("Final Answer")
            st.write(result["answer"])

            # st.subheader("Sources")
            # for s in result["sources"]:
            #     st.write(f"• {s}")

            if read_aloud:
                audio_path = synthesize_to_wav(result["answer"])
                st.audio(audio_path)
