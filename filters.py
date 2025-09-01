from typing import Tuple, List
from openai import OpenAI

PROFANITY = {
    # light example list; expand if your course provides a stricter lexicon
    "damn", "hell", "bastard", "idiot", "stupid",
    # add slurs or stronger words per policy if required by your assignment
}

class SafetyFilter:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else None

    def check(self, text: str) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        lowered = text.lower()
        prof_hits = [w for w in PROFANITY if w in lowered]
        if prof_hits:
            reasons.append(f"profanity detected: {', '.join(sorted(set(prof_hits)))}")

        # moderation pass if key is present
        if self.client:
            try:
                mod = self.client.moderations.create(
                    model="omni-moderation-latest", input=text
                )
                flagged = getattr(mod, "results", [])[0].flagged
                if flagged:
                    reasons.append("OpenAI moderation flagged the text.")
            except Exception:
                # Fail-open to the heuristic only
                pass

        return (len(reasons) > 0, reasons)
