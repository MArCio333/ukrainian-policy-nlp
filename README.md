# ua-policy-nlp

NLP pipeline for topic modeling and clustering of Ukrainian-language policy documents using Stanza, TF-IDF, and LDA.

---

## What it does

This pipeline processes a corpus of Ukrainian-language PDF and DOCX documents and produces:

- **Topic modeling** — LDA-based extraction of latent topics across the corpus
- **Document clustering** — K-Means grouping of documents by content similarity
- **Key phrase extraction** — TF-IDF ranked n-grams (bigrams and trigrams)
- **Style metrics** — average sentence length and modal verb frequency per document

It was built to analyze policy documents from Ukrainian government sources, where accurate morphological analysis of Ukrainian is essential.

---

## Why these tools

**Stanza over spaCy** — spaCy's Ukrainian language support produced poor lemmatization results in testing. Stanza's Ukrainian model handled morphological complexity significantly better, which matters for TF-IDF quality downstream.

**LDA over BERTopic** — the corpus size was too small for reliable dense embeddings. LDA performs well at this scale and is interpretable without requiring a large document count.

**MIN_DF=3** — set empirically. Lower values retained noise tokens without contributing analytically useful terms given the total corpus size. Higher values discarded relevant low-frequency domain terms.

**Cluster interpretation** — clusters were analyzed and labeled manually after generation, not automatically.

---

## Requirements

```
stanza==1.11.1
pdfplumber==0.11.9
scikit-learn==1.8.0
numpy==2.4.2
scipy==1.17.1
tqdm==4.67.3
docx==0.2.4
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The Ukrainian Stanza model will be downloaded automatically on first run if not already present.

---

## Usage

```bash
python ua_policy_nlp.py --folder /path/to/your/documents
```

The script processes all `.pdf` and `.docx` files found in the folder. Output is saved as `results.json` in the same folder and also printed to stdout.

### Parameters (edit at top of script)

| Parameter | Default | Description |
|---|---|---|
| `N_TOPICS` | 10 | Number of LDA topics |
| `N_CLUSTERS` | 10 | Number of K-Means clusters |
| `N_TOP_WORDS` | 40 | Words per topic |
| `MIN_DF` | 3 | Minimum document frequency for TF-IDF |
| `MAX_DF` | 0.85 | Maximum document frequency for TF-IDF |
| `MAX_FEATURES` | 5000 | Maximum vocabulary size for vectorizer |

---

## Stopwords

The pipeline applies three layers of stopword filtering:

- **Ukrainian function words** — pronouns, conjunctions, prepositions, auxiliary verbs
- **English function words** — for mixed-language documents
- **Legal boilerplate** — high-frequency terms specific to Ukrainian administrative documents (закон, стаття, кабінет, міністр, etc.) that appear across all documents and carry no discriminative value

---

## Output structure

```json
{
  "filenames": ["doc1.pdf", "doc2.docx"],
  "topics": [["term1", "term2", ...], ...],
  "clusters": [0, 1, 0, 2, ...],
  "phrases": ["key phrase one", "key phrase two", ...],
  "style": {
    "avg_sentence_length": 14.2,
    "total_modal_verbs": 87
  }
}
```

- `topics` — list of N_TOPICS topic word lists, ordered by weight
- `clusters` — cluster assignment per document, aligned with `filenames`
- `phrases` — top 50 bigrams and trigrams by TF-IDF score across corpus
- `style` — corpus-level style indicators; modal verbs tracked are повинен, має, може, слід, необхідно

See `sample_output.json` for a real example with annotated topic labels.

---

## Notes

- Both `.pdf` and `.docx` files are processed; other formats are ignored
- Documents that fail to extract text are skipped with a warning
- Text chunks over 20,000 characters are skipped to avoid Stanza memory issues on very large pages
- The pipeline is CPU-only (`use_gpu=False`)
- Progress bars show processing status for each pipeline stage

---

## License

MIT
