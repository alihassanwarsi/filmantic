import spacy

KEEP = {"NOUN", "PROPN", "VERB", "ADJ"}

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def lemmatize(texts):
    """Take a list of texts and return a list of cleaned texts made of lemmas"""

    cleaned = []

    for doc in nlp.pipe(texts):
        lemmas = [
            token.lemma_.lower() for token in doc
            if token.pos_ in KEEP and token.is_alpha and not token.is_stop
        ]

        cleaned.append(" ".join(lemmas))

    return cleaned

if __name__ == "__main__":
    from .data import load_movies

    df = load_movies()
    print(df["overview"].iloc[0])
    print()
    print(lemmatize(df["overview"].head(1).tolist())[0])