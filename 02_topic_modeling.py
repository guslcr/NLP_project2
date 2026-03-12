"""
PROJECT – Part 2: Topic Modeling (LDA + NMF)
=============================================
• LDA topic modeling on French reviews
• NMF as comparison
• Manual topic labeling for insurance domain
• Produces: data_with_topics.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

# ── Load clean data ───────────────────────────────────────────────────────────
df = pd.read_csv("data_clean.csv")
corpus = df["avis_clean_fr"].fillna("").tolist()
print(f"Loaded {len(df):,} rows")

# =============================================================================
# INSURANCE TOPIC LABELS (domain knowledge)
# =============================================================================
INSURANCE_TOPICS = {
    0: "Customer Service",
    1: "Claims Processing",
    2: "Pricing & Contracts",
    3: "Coverage & Guarantees",
    4: "Cancellation",
    5: "Online / Digital",
    6: "Auto Insurance",
    7: "Health Insurance",
}

TOPIC_KEYWORDS_EXPECTED = {
    "Customer Service":   ["conseiller","service","telephone","appel","contact","accueil","equipe","reponse"],
    "Claims Processing":  ["sinistre","remboursement","dossier","delai","indemnisation","declaration","prise","charge"],
    "Pricing & Contracts":["prix","tarif","cotisation","prime","contrat","augmentation","cout","economique"],
    "Coverage & Guarantees":["garantie","couverture","assurance","risque","protection","responsabilite"],
    "Cancellation":       ["resiliation","resilier","echéance","annulation","renouvellement","loi","hamon"],
    "Online / Digital":   ["application","site","espace","ligne","interface","numerique","connexion"],
    "Auto Insurance":     ["vehicule","voiture","accident","collision","conducteur","permis","km"],
    "Health Insurance":   ["sante","medecin","consultation","hospitalisation","optique","dentaire","mutuelle"],
}

N_TOPICS = 8

# =============================================================================
# LDA
# =============================================================================
print("Training LDA model...")
cv = CountVectorizer(max_features=5000, min_df=5, max_df=0.85)
X_cv = cv.fit_transform(corpus)
feature_names = cv.get_feature_names_out()

lda = LatentDirichletAllocation(
    n_components=N_TOPICS, max_iter=20, learning_method="online",
    random_state=42, n_jobs=-1
)
lda.fit(X_cv)
doc_topics_lda = lda.transform(X_cv)
df["lda_topic"]       = doc_topics_lda.argmax(axis=1)
df["lda_topic_label"] = df["lda_topic"].map(INSURANCE_TOPICS)
df["lda_confidence"]  = doc_topics_lda.max(axis=1)

# LDA topic-word distributions (top 10 words per topic)
lda_topics = {}
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    lda_topics[idx] = top_words
    print(f"LDA Topic {idx} [{INSURANCE_TOPICS.get(idx,'?')}]: {', '.join(top_words)}")

# =============================================================================
# NMF
# =============================================================================
print("\nTraining NMF model...")
tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.85)
X_tfidf = tfidf.fit_transform(corpus)
feat_tfidf = tfidf.get_feature_names_out()

nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=400)
nmf.fit(X_tfidf)
doc_topics_nmf = nmf.transform(X_tfidf)
df["nmf_topic"]  = doc_topics_nmf.argmax(axis=1)
df["nmf_topic_label"] = df["nmf_topic"].map(INSURANCE_TOPICS)

nmf_topics = {}
for idx, topic in enumerate(nmf.components_):
    top_words = [feat_tfidf[i] for i in topic.argsort()[:-11:-1]]
    nmf_topics[idx] = top_words
    print(f"NMF Topic {idx} [{INSURANCE_TOPICS.get(idx,'?')}]: {', '.join(top_words)}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
fig = plt.figure(figsize=(20, 22))
fig.suptitle("Part 2 – Topic Modeling (LDA + NMF)", fontsize=15, fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 1, N_TOPICS))

# ── LDA topic-word heatmap ────────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
top_n = 10
lda_mat = np.zeros((N_TOPICS, top_n))
top_global_words = []
for i, (idx, topic) in enumerate(enumerate(lda.components_)):
    top_ids = topic.argsort()[:-top_n-1:-1]
    if i == 0: top_global_words = [feature_names[j] for j in top_ids]
    lda_mat[i] = topic[top_ids] / topic.sum()

im = ax1.imshow(lda_mat, cmap="YlOrRd", aspect="auto")
ax1.set_xticks(range(top_n))
ax1.set_xticklabels(top_global_words, rotation=45, ha="right", fontsize=8)
ax1.set_yticks(range(N_TOPICS))
ax1.set_yticklabels([INSURANCE_TOPICS.get(i, f"T{i}") for i in range(N_TOPICS)], fontsize=8)
ax1.set_title("LDA Topic-Word Distribution")
plt.colorbar(im, ax=ax1)

# ── Document-topic distribution (LDA) ────────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
topic_counts = df["lda_topic_label"].value_counts()
bars = ax2.bar(range(len(topic_counts)), topic_counts.values,
               color=[colors[i] for i in range(len(topic_counts))], edgecolor="white")
ax2.set_xticks(range(len(topic_counts)))
ax2.set_xticklabels(topic_counts.index, rotation=35, ha="right", fontsize=8)
ax2.set_title("Document Count per LDA Topic")
ax2.set_ylabel("# Reviews")
for bar, val in zip(bars, topic_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{val:,}", ha="center", fontsize=7)

# ── Average rating per topic ─────────────────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
topic_rating = df.groupby("lda_topic_label")["note"].mean().sort_values()
bars3 = ax3.barh(topic_rating.index, topic_rating.values,
                 color=["#e74c3c" if v < 2.5 else "#f39c12" if v < 3.5 else "#2ecc71"
                        for v in topic_rating.values], edgecolor="white")
ax3.axvline(df["note"].mean(), color="navy", linestyle="--", linewidth=1.2)
ax3.set_xlim(1, 5)
ax3.set_title("Average Rating by LDA Topic")
ax3.set_xlabel("Avg Rating")
for bar, val in zip(bars3, topic_rating.values):
    ax3.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=9)

# ── Top words per LDA topic (4 subplots) ─────────────────────────────────────
for i in range(4):
    ax = fig.add_subplot(3, 4, 5+i)  # row 2
    words_i = lda_topics[i][:10]
    freqs_i = lda.components_[i][
        [list(feature_names).index(w) for w in words_i]
    ]
    freqs_i = freqs_i / freqs_i.sum()
    ax.barh(words_i[::-1], freqs_i[::-1], color=colors[i], edgecolor="white")
    ax.set_title(f"LDA T{i}: {INSURANCE_TOPICS.get(i,'?')}", fontsize=9)
    ax.set_xlabel("Norm. weight")
    ax.tick_params(axis="y", labelsize=8)

# ── Top words per NMF topic (4 subplots) ─────────────────────────────────────
for i in range(4):
    ax = fig.add_subplot(3, 4, 9+i)  # row 3
    words_i = nmf_topics[i][:10]
    freqs_i = nmf.components_[i][
        [list(feat_tfidf).index(w) for w in words_i]
    ]
    freqs_i = freqs_i / freqs_i.sum()
    ax.barh(words_i[::-1], freqs_i[::-1], color=colors[i+4], edgecolor="white")
    ax.set_title(f"NMF T{i}: {INSURANCE_TOPICS.get(i,'?')}", fontsize=9)
    ax.set_xlabel("Norm. weight")
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("02_topic_modeling.png", bbox_inches="tight", dpi=110)
plt.close()
print("\nSaved: 02_topic_modeling.png")

# =============================================================================
# TOPIC × SENTIMENT CROSS TABLE
# =============================================================================
ct = pd.crosstab(df["lda_topic_label"], df["sentiment"], normalize="index") * 100
print("\nTopic × Sentiment (%):")
print(ct.round(1).to_string())

# Save
df.to_csv("data_with_topics.csv", index=False)
print(f"\nSaved: data_with_topics.csv")
