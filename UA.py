#!/usr/bin/env python3

import time
from pathlib import Path
from typing import List, Dict

# Progress
from tqdm import tqdm

# NLP
import stanza

# File extraction
import pdfplumber
from docx import Document

# ML
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# -----------------------------
# CONFIG
# -----------------------------
N_TOPICS = 5
N_CLUSTERS = 5
N_TOP_WORDS = 20
MIN_DF = 5
MAX_DF = 0.75
MAX_FEATURES = 5000

# -----------------------------
# STOPWORDS
# -----------------------------
UK_STOPWORDS = {
    "це","що","який","яка","яке","та","або","для","на","у","з","до",
    "від","за","про","із","як","не","так","бути","є","був","була"
}

EN_STOPWORDS = {
    "the","and","is","in","to","of","for","on","with","as","by","that"
}

LEGAL_STOPWORDS = {
    "закон","україна","стаття","пункт","відповідно",
    "кабінет","міністр","рішення","орган"
}

STOPWORDS = UK_STOPWORDS | EN_STOPWORDS | LEGAL_STOPWORDS

# -----------------------------
# LOAD STANZA
# -----------------------------
def load_nlp():
    try:
        return stanza.Pipeline("uk", processors="tokenize,lemma", verbose=False)
    except:
        print("⬇️ Downloading Stanza model...")
        stanza.download("uk")
        return stanza.Pipeline("uk", processors="tokenize,lemma", verbose=False)

nlp = load_nlp()

# -----------------------------
# FILE EXTRACTION
# -----------------------------
def extract_text_from_pdf(path: str) -> str:
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text.append(content)
    except Exception as e:
        print(f"PDF error {path}: {e}")
    return "\n".join(text)


def extract_text_from_docx(path: str) -> str:
    text = []
    try:
        doc = Document(path)
        for para in doc.paragraphs:
            if para.text:
                text.append(para.text)
    except Exception as e:
        print(f"DOCX error {path}: {e}")
    return "\n".join(text)


def load_documents(folder_path: str):
    texts, filenames = [], []

    files = [p for p in Path(folder_path).rglob("*") if p.is_file()]

    for path in tqdm(files, desc="📂 Loading files"):
        ext = path.suffix.lower()

        if ext == ".pdf":
            text = extract_text_from_pdf(str(path))
        elif ext == ".docx":
            text = extract_text_from_docx(str(path))
        else:
            continue

        if text.strip():
            texts.append(text)
            filenames.append(str(path))

    print(f"Loaded {len(texts)} documents")
    return texts, filenames

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess(text: str) -> str:
    tokens = []

    for chunk in text.split("\n"):
        chunk = chunk.strip()

        if not chunk or len(chunk) > 20000:
            continue

        try:
            doc = nlp(chunk.lower())
        except:
            continue

        for sentence in doc.sentences:
            for word in sentence.words:
                lemma = word.lemma

                if not lemma:
                    continue
                if not lemma.isalpha():
                    continue
                if lemma in STOPWORDS:
                    continue
                if len(lemma) < 3:
                    continue

                tokens.append(lemma)

    return " ".join(tokens)


def preprocess_corpus(texts: List[str]) -> List[str]:
    processed = []
    for text in tqdm(texts, desc="🧹 Preprocessing"):
        processed.append(preprocess(text))
    return processed

# -----------------------------
# VECTORIZERS
# -----------------------------
def compute_tfidf(texts: List[str]):
    vectorizer = TfidfVectorizer(
        max_df=MAX_DF,
        min_df=MIN_DF,
        ngram_range=(1, 3),
        max_features=MAX_FEATURES
    )
    return vectorizer.fit_transform(texts), vectorizer


def compute_counts(texts: List[str]):
    vectorizer = CountVectorizer(
        max_df=MAX_DF,
        min_df=MIN_DF,
        ngram_range=(1, 2),
        max_features=MAX_FEATURES
    )
    return vectorizer.fit_transform(texts), vectorizer

# -----------------------------
# TOPICS
# -----------------------------
def extract_topics(X_counts, vectorizer):
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42
    )
    lda.fit(X_counts)

    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic in lda.components_:
        words = [
            feature_names[i]
            for i in topic.argsort()[-N_TOP_WORDS:]
        ]
        topics.append(words)

    return topics

# -----------------------------
# CLUSTERING
# -----------------------------
def cluster_documents(X):
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=20
    )
    return model.fit_predict(X)

# -----------------------------
# PHRASES
# -----------------------------
def extract_phrases(texts: List[str], top_n=30):
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 3),
        max_df=0.7,
        min_df=5,
        max_features=3000
    )

    X = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()

    scores = X.sum(axis=0).A1
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)

    return [p for p, _ in ranked[:top_n]]

# -----------------------------
# STYLE
# -----------------------------
def compute_style_metrics(text: str) -> Dict:
    try:
        doc = nlp(text)
    except:
        return {"avg_sentence_length": 0, "modal_verb_count": 0}

    lengths = [len(s.words) for s in doc.sentences]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    modal_verbs = {"повинен","має","може","слід","необхідно"}

    modal_count = sum(
        1 for s in doc.sentences for w in s.words if w.lemma in modal_verbs
    )

    return {
        "avg_sentence_length": avg_len,
        "modal_verb_count": modal_count
    }


def corpus_style(texts: List[str]) -> Dict:
    metrics = []

    for text in tqdm(texts, desc="Style analysis"):
        metrics.append(compute_style_metrics(text))

    return {
        "avg_sentence_length": sum(m["avg_sentence_length"] for m in metrics) / len(metrics),
        "total_modal_verbs": sum(m["modal_verb_count"] for m in metrics)
    }

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def analyze_corpus(folder_path: str) -> Dict:
    print("\n Starting analysis...\n")

    start = time.time()

    print(" Step 1: Loading")
    texts, filenames = load_documents(folder_path)

    print("\n Step 2: Preprocessing")
    processed = preprocess_corpus(texts)

    print("\n Step 3: Vectorization")
    X_tfidf, tfidf_vec = compute_tfidf(processed)
    X_counts, count_vec = compute_counts(processed)

    print("\n Step 4: Topic modeling")
    topics = extract_topics(X_counts, count_vec)

    print("\n Step 5: Clustering")
    clusters = cluster_documents(X_tfidf)

    print("\n Step 6: Phrase extraction")
    phrases = extract_phrases(processed)

    print("\n Step 7: Style analysis")
    style = corpus_style(texts)

    print(f"\n Done in {time.time() - start:.2f} seconds\n")

    return {
        "filenames": filenames,
        "topics": topics,
        "clusters": clusters.tolist(),
        "phrases": phrases,
        "style": style
    }

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Ukrainian text analysis pipeline")
    parser.add_argument("folder", type=str)

    args = parser.parse_args()

    results = analyze_corpus(args.folder)

    print(json.dumps(results, indent=2, ensure_ascii=False))
