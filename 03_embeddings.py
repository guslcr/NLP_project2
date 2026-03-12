"""
PROJECT – Part 3: Word Embeddings (Word2Vec + GloVe-style)
===========================================================
• Train Word2Vec on the corpus
• Visualize with PCA/UMAP
• Cosine & Euclidean distance functions
• Semantic search
• Produces: word2vec.model, embeddings_2d.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, KeyedVectors
import warnings
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_with_topics.csv")
corpus_clean = df["avis_clean_fr"].fillna("").tolist()

# Tokenize
sentences = [text.split() for text in corpus_clean if len(text.split()) >= 3]
print(f"Training Word2Vec on {len(sentences):,} sentences...")

# =============================================================================
# WORD2VEC TRAINING
# =============================================================================
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,       # embedding dimension
    window=5,              # context window
    min_count=10,          # min frequency
    workers=4,
    sg=1,                  # Skip-gram (sg=1) vs CBOW (sg=0)
    epochs=15,
    seed=42
)
w2v_model.save("word2vec.model")
vocab_size = len(w2v_model.wv)
print(f"Vocabulary size: {vocab_size:,} words")
print(f"Embedding dimension: {w2v_model.vector_size}")

wv = w2v_model.wv

# =============================================================================
# DISTANCE FUNCTIONS
# =============================================================================

def cosine_distance(w1: str, w2: str) -> float:
    """Cosine distance between two words (1 - cosine similarity)."""
    if w1 not in wv or w2 not in wv:
        return None
    v1, v2 = wv[w1].reshape(1,-1), wv[w2].reshape(1,-1)
    return float(1 - cosine_similarity(v1, v2)[0,0])

def euclidean_distance(w1: str, w2: str) -> float:
    """Euclidean distance between two words."""
    if w1 not in wv or w2 not in wv:
        return None
    return float(np.linalg.norm(wv[w1] - wv[w2]))

def semantic_search(query_word: str, topn: int = 10) -> list:
    """Find top-N semantically similar words."""
    if query_word not in wv:
        return []
    return wv.most_similar(query_word, topn=topn)

# =============================================================================
# EXAMPLE QUERIES
# =============================================================================
QUERY_WORDS = ["remboursement","sinistre","conseiller","resiliation","prix",
               "satisfait","probleme","service","garantie","contrat"]

print("\n── Similar words examples ──────────────────────────────────────────")
for qw in ["remboursement","sinistre","satisfait","probleme"]:
    if qw in wv:
        sims = semantic_search(qw, topn=6)
        print(f"\n'{qw}' → {[(w, round(s,3)) for w,s in sims]}")

print("\n── Distance examples ────────────────────────────────────────────────")
pairs = [("remboursement","sinistre"),("satisfait","content"),
         ("probleme","difficulte"),("prix","tarif"),("annulation","resiliation")]
for w1, w2 in pairs:
    cd = cosine_distance(w1, w2)
    ed = euclidean_distance(w1, w2)
    if cd is not None:
        print(f"  {w1} ↔ {w2}: cosine_dist={cd:.4f}, euclidean_dist={ed:.4f}")

# =============================================================================
# PCA VISUALIZATION
# =============================================================================
# Select representative words for visualization
WORD_GROUPS = {
    "Positif":      ["satisfait","excellent","parfait","content","super","recommande","merci","bravo"],
    "Négatif":      ["probleme","mauvais","terrible","nul","arnaque","honte","fuir","decevant"],
    "Sinistre":     ["sinistre","accident","dommage","reparation","expertise","remboursement"],
    "Service":      ["conseiller","service","telephone","appel","contact","equipe","reponse"],
    "Contrat":      ["contrat","resiliation","annulation","echeance","prime","cotisation"],
    "Couverture":   ["garantie","couverture","protection","risque","franchise","assure"],
}

all_words, all_groups, all_vectors = [], [], []
for group, words in WORD_GROUPS.items():
    for w in words:
        if w in wv:
            all_words.append(w)
            all_groups.append(group)
            all_vectors.append(wv[w])

vectors_np = np.array(all_vectors)

# PCA 2D
pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(vectors_np)

# PCA 3D for Tensorboard-style data
pca3 = PCA(n_components=3, random_state=42)
coords_3d = pca3.fit_transform(vectors_np)

# Save embeddings for Tensorboard
all_vectors_df = pd.DataFrame(
    coords_3d, columns=["x","y","z"]
)
all_vectors_df["word"]  = all_words
all_vectors_df["group"] = all_groups
all_vectors_df.to_csv("embeddings_3d.tsv", sep="\t", index=False)

# Also save full vocab embeddings for Tensorboard
vocab_words = list(wv.key_to_index.keys())[:2000]
full_embs = np.array([wv[w] for w in vocab_words])
pca_full = PCA(n_components=2, random_state=42).fit_transform(full_embs)
emb_df = pd.DataFrame(pca_full, columns=["x","y"])
emb_df["word"] = vocab_words
emb_df.to_csv("embeddings_2d.csv", index=False)
print(f"\nSaved embeddings for {len(vocab_words)} words → embeddings_2d.csv")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
GROUP_COLORS = {
    "Positif": "#2ecc71", "Négatif": "#e74c3c", "Sinistre": "#e67e22",
    "Service": "#3498db", "Contrat": "#9b59b6", "Couverture": "#1abc9c"
}

fig = plt.figure(figsize=(20, 20))
fig.suptitle("Part 3 – Word Embeddings (Word2Vec)", fontsize=15, fontweight="bold")

# ── Plot 1: PCA 2D scatter ────────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
for group in WORD_GROUPS:
    mask = [g == group for g in all_groups]
    xs = coords_2d[mask, 0]
    ys = coords_2d[mask, 1]
    ax1.scatter(xs, ys, label=group, color=GROUP_COLORS[group], s=80, zorder=3, edgecolors="white")
    for word, x, y in zip(np.array(all_words)[mask], xs, ys):
        ax1.annotate(word, (x, y), fontsize=7, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")
ax1.legend(fontsize=7, markerscale=0.8)
ax1.set_title(f"Word2Vec PCA 2D (var={sum(pca.explained_variance_ratio_)*100:.1f}%)")
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
ax1.grid(alpha=0.3)

# ── Plot 2: Full vocab PCA 2D (density) ──────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.scatter(pca_full[:, 0], pca_full[:, 1], alpha=0.08, s=5, color="#2980b9")
for _, row in emb_df.sample(60, random_state=42).iterrows():
    ax2.annotate(row["word"], (row["x"], row["y"]), fontsize=6, alpha=0.7)
ax2.set_title(f"Full Vocabulary PCA 2D (top 2000 words)")
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

# ── Plot 3: Cosine similarity heatmap ────────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
sample_words = [w for w in ["remboursement","sinistre","conseiller","resiliation",
                              "prix","satisfait","probleme","garantie","contrat",
                              "telephone","application","accident"] if w in wv][:10]
sim_mat = np.zeros((len(sample_words), len(sample_words)))
for i, w1 in enumerate(sample_words):
    for j, w2 in enumerate(sample_words):
        sim_mat[i,j] = float(cosine_similarity(
            wv[w1].reshape(1,-1), wv[w2].reshape(1,-1))[0,0])
im3 = ax3.imshow(sim_mat, cmap="RdYlGn", vmin=-1, vmax=1)
ax3.set_xticks(range(len(sample_words)))
ax3.set_yticks(range(len(sample_words)))
ax3.set_xticklabels(sample_words, rotation=45, ha="right", fontsize=8)
ax3.set_yticklabels(sample_words, fontsize=8)
for i in range(len(sample_words)):
    for j in range(len(sample_words)):
        ax3.text(j, i, f"{sim_mat[i,j]:.2f}", ha="center", va="center", fontsize=6)
ax3.set_title("Cosine Similarity Heatmap")
plt.colorbar(im3, ax=ax3)

# ── Plots 4-6: most_similar bar charts for 3 query words ─────────────────────
for plot_i, qw in enumerate(["remboursement","sinistre","satisfait"]):
    ax = fig.add_subplot(3, 3, 4+plot_i)
    if qw in wv:
        sims = wv.most_similar(qw, topn=10)
        words_sim, scores = zip(*sims)
        ax.barh(words_sim[::-1], scores[::-1],
                color=["#3498db" if s > 0.5 else "#95a5a6" for s in scores[::-1]],
                edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_title(f"Most similar to '{qw}'")
        ax.set_xlabel("Cosine similarity")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8)

# ── Plot 7: Word analogy evaluation ──────────────────────────────────────────
ax7 = fig.add_subplot(3, 3, 7)
analogies = []
try:
    # A:B :: C:? style
    analogy_tests = [
        ("bon","mauvais","bien"),
        ("telephone","conseiller","internet"),
        ("satisfait","content","insatisfait"),
    ]
    for a, b, c in analogy_tests:
        if all(w in wv for w in [a,b,c]):
            result = wv.most_similar(positive=[b,c], negative=[a], topn=3)
            analogies.append(f"{a}:{b}::{c}:{result[0][0]} ({result[0][1]:.2f})")
except: pass

ax7.axis("off")
text_lines = ["Word2Vec Model Summary", "─"*35,
              f"Vocabulary: {vocab_size:,} words",
              f"Dimensions: {w2v_model.vector_size}",
              f"Window: {w2v_model.window}",
              f"Algorithm: Skip-gram",
              f"Training sentences: {len(sentences):,}",
              "", "Analogy examples:"] + analogies
for i, line in enumerate(text_lines):
    weight = "bold" if i < 2 else "normal"
    ax7.text(0.05, 0.95 - i*0.07, line, transform=ax7.transAxes,
             fontsize=9, verticalalignment="top", fontweight=weight,
             fontfamily="monospace")
ax7.set_title("Model Summary & Analogies")

# ── Plot 8: PCA explained variance ───────────────────────────────────────────
ax8 = fig.add_subplot(3, 3, 8)
pca_full_model = PCA(n_components=20, random_state=42)
pca_full_model.fit(full_embs)
cumvar = np.cumsum(pca_full_model.explained_variance_ratio_) * 100
ax8.plot(range(1, 21), cumvar, "bo-", linewidth=2, markersize=5)
ax8.fill_between(range(1, 21), cumvar, alpha=0.15)
ax8.axhline(80, color="red", linestyle="--", label="80% threshold")
ax8.set_title("PCA Explained Variance (Word2Vec)")
ax8.set_xlabel("Number of components")
ax8.set_ylabel("Cumulative variance (%)")
ax8.legend(fontsize=9)
ax8.grid(alpha=0.3)

# ── Plot 9: Semantic search demo ─────────────────────────────────────────────
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
search_demos = []
for qw in ["probleme","remboursement","annulation"]:
    if qw in wv:
        results = semantic_search(qw, topn=5)
        search_demos.append(f"Query: '{qw}'")
        for w, s in results:
            search_demos.append(f"  → {w:<18} sim={s:.3f}")
        search_demos.append("")
ax9.text(0.02, 0.98, "\n".join(search_demos), transform=ax9.transAxes,
         fontsize=8, verticalalignment="top", fontfamily="monospace")
ax9.set_title("Semantic Search Demo")

plt.tight_layout()
plt.savefig("03_embeddings.png", bbox_inches="tight", dpi=110)
plt.close()
print("Saved: 03_embeddings.png")

# =============================================================================
# TENSORBOARD VECTORS FILE (vectors.tsv + metadata.tsv)
# =============================================================================
sample_size = 1000
sample_words = vocab_words[:sample_size]
with open("tensorboard_vectors.tsv", "w") as fv, \
     open("tensorboard_metadata.tsv", "w") as fm:
    fm.write("word\n")
    for w in sample_words:
        vec_str = "\t".join(str(round(float(v), 6)) for v in wv[w])
        fv.write(vec_str + "\n")
        fm.write(w + "\n")
print(f"Tensorboard files saved (top {sample_size} words)")
print("\nTo launch Tensorboard:")
print("  tensorboard --logdir=. --port=6006")
print("  → 'Projector' tab → load tensorboard_vectors.tsv + tensorboard_metadata.tsv")
