from typing import Optional, List, Dict
from openai import OpenAI

class HyDEGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str) -> str:
        system_prompt = (
            "You are a helpful assistant for a book-summary librarian. "
            "Generate a short, plausible hypothetical answer to the user's question "
            "as if it could be found in a classic novel summary database. "
            "Keep it neutral and non-fantastical; stay within book summaries."
        )

        few_shots: List[Dict[str, str]] = [
            {"role": "user", "content": "What is the central conflict in 'Brave New World'?"},
            {"role": "assistant", "content": "The tension between engineered social stability and individual freedom, "
                                             "as characters struggle against conditioning and the costs of a controlled utopia."},
        ]

        messages = [{"role": "system", "content": system_prompt}] + few_shots + [
            {"role": "user", "content": question}
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.6,
            max_tokens=160
        )
        return resp.choices[0].message.content.strip()
