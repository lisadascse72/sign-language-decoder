# 🔮 Predict sentence from letter sequence
def predict_sentence_from_letters(letter_sequence: str) -> str:
    """
    Predicts the most probable English word or sentence from a sequence of letters using Gemini API.
    """
    if not letter_sequence.strip():
        return "⚠️ Empty letter sequence. Please try again."

    try:
        prompt = (
            f"You are an intelligent assistant helping decode sign language letters.\n"
            f"Given the sequence of letters: '{letter_sequence}', "
            "suggest the most probable English word or sentence.\n"
            "If it's not a real word, guess the closest meaningful phrase."
        )
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "⚠️ No response from Gemini."
    except Exception as e:
        if "429" in str(e):
            return (
                "⚠️ Rate limit exceeded: You've hit the free-tier quota. "
                "Please wait a few minutes and try again.\n"
                "Visit [Gemini API Quota Docs](https://ai.google.dev/gemini-api/docs/rate-limits) for more info."
            )
        return f"⚠️ Gemini Error: {e}"
