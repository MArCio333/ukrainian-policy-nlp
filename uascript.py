
import os
from typing import List, Dict

# NLP 
import stanza

# PDF extraction
import pdfplumber

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# -----------------------------
# CONFIG
# -----------------------------
N_TOPICS = 10
N_CLUSTERS = 10
N_TOP_WORDS = 40
MIN_DF = 3
MAX_DF = 0.85

# -----------------------------
# LOAD UKRAINIAN MODEL (STANZA)
# -----------------------------
try:
    nlp = stanza.Pipeline(
        "uk",
        processors="tokenize,lemma",
        use_gpu=False,
        verbose=False
    )
except:
    print("⬇️ Downloading Ukrainian model...")
    stanza.download("uk")
    nlp = stanza.Pipeline(
        "uk",
        processors="tokenize,lemma",
        use_gpu=False,
        verbose=False
    )

# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
    return text


def load_pdfs(folder_path: str):
    texts = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            text = extract_text_from_pdf(full_path)

            if text.strip():
                texts.append(text)
                filenames.append(file)

    return texts, filenames


# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess(text: str) -> str:
    tokens = []

    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue

        # prevent crashes on huge chunks
        if len(chunk) > 20000:
            continue

        try:
            doc = nlp(chunk.lower())
        except Exception as e:
            print("⚠️ Skipping chunk:", e)
            continue

        for sentence in doc.sentences:
            for word in sentence.words:
                if not word.text.isalpha():
                    continue

                if word.lemma:
                    tokens.append(word.lemma)

    return " ".join(tokens)


def preprocess_corpus(texts: List[str]) -> List[str]:
    return [preprocess(t) for t in texts]


# -----------------------------
# TF-IDF
# -----------------------------
def compute_tfidf(texts: List[str]):
    vectorizer = TfidfVectorizer(
        max_df=MAX_DF,
        min_df=MIN_DF,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)
    return X, vectorizer


# -----------------------------
# TOPIC MODELING
# -----------------------------
def extract_topics(X, vectorizer):
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42
    )
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic in lda.components_:
        top_words = [
            feature_names[i]
            for i in topic.argsort()[-N_TOP_WORDS:]
        ]
        topics.append(top_words)

    return topics


# -----------------------------
# CLUSTERING
# -----------------------------
def cluster_documents(X):
    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    model.fit(X)
    return model.labels_


# -----------------------------
# PHRASE EXTRACTION
# -----------------------------
def extract_phrases(texts: List[str], top_n: int = 50):
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 3),
        max_df=MAX_DF,
        min_df=MIN_DF
    )

    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    scores = X.sum(axis=0).A1
    pairs = list(zip(feature_names, scores))

    pairs.sort(key=lambda x: x[1], reverse=True)

    return [p for p, _ in pairs[:top_n]]


# -----------------------------
# STYLE ANALYSIS 
# -----------------------------
def compute_style_metrics(text: str) -> Dict:
    doc = nlp(text)

    sentence_lengths = [
        len(sentence.words) for sentence in doc.sentences
    ]

    avg_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    modal_verbs = {"повинен", "має", "може", "слід", "необхідно"}
    modal_count = sum(
        1
        for sentence in doc.sentences
        for word in sentence.words
        if word.lemma in modal_verbs
    )

    return {
        "avg_sentence_length": avg_len,
        "modal_verb_count": modal_count
    }


def corpus_style(texts: List[str]) -> Dict:
    metrics = [compute_style_metrics(t) for t in texts]

    return {
        "avg_sentence_length": sum(m["avg_sentence_length"] for m in metrics) / len(metrics),
        "total_modal_verbs": sum(m["modal_verb_count"] for m in metrics)
    }


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def analyze_corpus(folder_path: str) -> Dict:
    print("Load PDF")
    texts, filenames = load_pdfs(folder_path)

    processed = preprocess_corpus(texts)

    print("copmuting TF-IDF")
    X, vectorizer = compute_tfidf(processed)

    print("Topics")
    topics = extract_topics(X, vectorizer)

    print("Cluster")
    clusters = cluster_documents(X)

    print("Phrases")
    phrases = extract_phrases(processed, top_n=50)

    print("Corpus")
    style = corpus_style(texts)

    return {
        "filenames": filenames,
        "topics": topics,
        "clusters": clusters.tolist(),
        "phrases": phrases,
        "style": style
    }

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import json

    folder_path = ""

    results = analyze_corpus(folder_path)

    print("\n--- РЕЗУЛЬТАТИ ---\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))