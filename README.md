# 🛡️ Insurance Review NLP Platform

An end-to-end NLP project on French insurance customer reviews — from data cleaning to a fully interactive Streamlit application.

## 📋 Project Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_data_cleaning.py` | Text cleaning, stopword removal, spelling correction, n-gram analysis |
| 2 | `02_topic_modeling.py` | LDA + NMF topic modeling (8 insurance topics) |
| 3 | `03_embeddings.py` | Word2Vec training, cosine/euclidean distances, semantic search, Tensorboard export |
| 4 | `04_supervised_learning.py` | TF-IDF + ML models for sentiment & star rating prediction |
| 5 | `app.py` | Interactive Streamlit application |

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/NLP_project2.git
cd NLP_project2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place `avis_clients.csv` in the root folder (not included in the repo — see [Data](#data) section below).

### 4. Run the pipeline
```bash
python 01_data_cleaning.py
python 02_topic_modeling.py
python 03_embeddings.py
python 04_supervised_learning.py
```

### 5. Launch the app
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 📊 App Features

| Tab | Feature |
|-----|---------|
| 🔮 Prediction | Star rating + sentiment + topic detection with probability breakdown |
| 📊 Insurer Analysis | Rankings, scatter plots, extractive summaries per insurer |
| 🔍 Review Search | TF-IDF semantic search with filters (insurer, product, stars) |
| 💡 Explanation | Token-level highlighting + LogReg feature importance (LIME-style) |
| 🔗 RAG | Retrieve top-K reviews by query → extractive summary |
| ❓ QA | Keyword-based question answering over reviews |

## 📈 Model Results

| Task | Model | Accuracy | F1 (weighted) |
|------|-------|----------|---------------|
| Sentiment (3-class) | TF-IDF + LogReg | 81.1% | 0.768 |
| Sentiment (3-class) | TF-IDF + LinearSVC | 80.6% | 0.766 |
| Sentiment (3-class) | W2V avg + LogReg | 81.0% | 0.763 |
| Star rating (5-class) | TF-IDF + LogReg | 51.2% | 0.472 |

## 🗂️ Data

The dataset (`avis_clients.csv`) contains ~34,000 French insurance reviews with columns:
`note`, `auteur`, `avis`, `assureur`, `produit`, `type`, `date_publication`, `avis_en`

> ⚠️ The dataset is **not included** in this repository. Place your own `avis_clients.csv` in the root folder before running the pipeline.

## 🧠 NLP Techniques Used

- Text cleaning & tokenization (custom FR/EN stopwords)
- N-gram analysis (unigrams, bigrams, trigrams)
- TF-IDF vectorization
- LDA & NMF topic modeling
- Word2Vec (Skip-gram, 100d) embeddings
- Cosine & Euclidean distance
- Semantic search
- Logistic Regression, LinearSVC, Random Forest classifiers
- Extractive summarization
- Tensorboard embedding visualization

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
gensim>=4.3.0
matplotlib>=3.7.0
plotly>=5.15.0
wordcloud>=1.9.0
scipy>=1.11.0
```

## 📁 Generated Files (after running pipeline)

| File | Description |
|------|-------------|
| `data_clean.csv` | Cleaned dataset with new text columns |
| `data_with_topics.csv` | Dataset with LDA/NMF topic assignments |
| `word2vec.model` | Trained Word2Vec model |
| `model_sentiment.pkl` | Best sentiment classifier |
| `model_stars.pkl` | Best star rating classifier |
| `embeddings_2d.csv` | PCA-reduced word embeddings |
| `tensorboard_vectors.tsv` | Tensorboard projector vectors |
| `tensorboard_metadata.tsv` | Tensorboard projector metadata |

## 🔭 Tensorboard Visualization

```bash
tensorboard --logdir=. --port=6006
```
Open [http://localhost:6006](http://localhost:6006) → **Projector** tab → load `tensorboard_vectors.tsv` + `tensorboard_metadata.tsv`

## 👤 Author

Built by Augustin Leclair as part of an NLP supervised & unsupervised learning project for ESILV A4 DIA.

## Streamlit app link

Visit the Streamlit app of the project here : https://insurancereviewapp.streamlit.app
