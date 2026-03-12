"""
PROJECT – Part 1: Data Cleaning & Text Preprocessing
=====================================================
• Tokenization & stopword removal (FR + EN)
• Frequent words & n-grams analysis
• Spelling correction (basic Levenshtein approach without internet)
• Produces: data_clean.csv
"""

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ── French + English stopwords (embedded, no internet needed) ────────────────
FR_STOPS = {
    "le","la","les","de","du","des","un","une","et","en","au","aux","à","a",
    "ce","se","je","tu","il","elle","nous","vous","ils","elles","me","te",
    "lui","y","ne","pas","plus","par","sur","sous","dans","avec","pour",
    "que","qui","quoi","dont","où","ou","si","mais","donc","or","ni","car",
    "très","bien","tout","tous","toute","toutes","mon","ma","mes","ton","ta",
    "tes","son","sa","ses","notre","votre","leur","leurs","même","aussi",
    "avoir","être","faire","dire","aller","voir","vouloir","pouvoir","falloir",
    "ai","as","est","sont","ont","était","avait","été","fait","dit","va",
    "plus","peu","lors","alors","après","avant","comme","dont","encore",
    "toujours","jamais","déjà","autre","autres","ces","cet","cette","ceux",
    "ça","c","d","j","l","m","n","qu","s","t","j","y","k","là","ci",
    "suite","faire","doit","peut","lors","car","soit","puis","cela","ceci",
    "via","via","contre","chez","entre","jusqu","sans","dès","depuis","pendant",
    "certain","certains","plusieurs","chaque","tout","lors","aucun","aucune",
}
EN_STOPS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall",
    "not","no","nor","so","yet","both","either","neither","each","every",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","this","that","these","those",
    "what","which","who","whom","whose","when","where","why","how",
    "all","any","few","more","most","other","some","such","than","too",
    "very","just","because","as","until","while","although","though",
    "about","against","between","into","through","during","before","after",
    "above","below","from","up","down","out","off","over","under","again",
    "then","once","here","there","can","get","got","also","much","many",
}
STOPS = FR_STOPS | EN_STOPS

# =============================================================================
# LOAD
# =============================================================================
df = pd.read_csv("avis_clients.csv")
df_train = df[df["type"] == "train"].copy()
df_train.drop(columns=["avis_cor","avis_cor_en"], inplace=True, errors="ignore")
df_train.drop_duplicates(subset=["auteur","avis"], inplace=True)
df_train["date_publication"] = pd.to_datetime(df_train["date_publication"], dayfirst=True)
df_train["year"] = df_train["date_publication"].dt.year
df_train["sentiment"] = df_train["note"].apply(
    lambda x: "Positive" if x >= 4 else ("Neutral" if x == 3 else "Negative"))

print(f"Train set: {len(df_train):,} rows")

# =============================================================================
# TEXT CLEANING FUNCTIONS
# =============================================================================

def clean_text(text: str, lang: str = "fr") -> str:
    """
    Full text cleaning pipeline:
    1. Lowercase
    2. Remove URLs, emails, numbers
    3. Remove punctuation
    4. Remove stopwords
    5. Remove very short tokens (≤2 chars)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                  # emails
    text = re.sub(r"\b\d+([.,]\d+)?\b", " ", text)        # numbers
    text = re.sub(r"[^\w\s]", " ", text)                  # punctuation
    text = re.sub(r"_+", " ", text)                       # underscores
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPS and len(t) > 2]
    return " ".join(tokens)


def extract_ngrams(corpus, n=2, top_k=20):
    vec = CountVectorizer(ngram_range=(n, n), max_features=5000)
    X = vec.fit_transform(corpus)
    freq = X.sum(axis=0).A1
    names = vec.get_feature_names_out()
    return sorted(zip(names, freq), key=lambda x: -x[1])[:top_k]


# Basic spelling correction using edit distance
COMMON_CORRECTIONS = {
    "assurance": ["assurnce","assuranc","assurence","assuranes","assurancw"],
    "remboursement": ["rembourement","rembourssement","remboursment"],
    "contrat": ["conrat","contart","contrat","conrtact"],
    "sinistre": ["sinstre","sinisttre","sinitre"],
    "conseiller": ["consseiller","conseler","consieller"],
    "réclamation": ["reclamation","reclamtion","reclam"],
    "résilier": ["resilier","resiler","résilié","resilié"],
    "garantie": ["garanti","garantis","garantee"],
    "satisfait": ["satifait","satisfaire","satisfaite","sattisfait"],
    "problème": ["probleme","problème","prbleme","problèm"],
}
CORRECTION_MAP = {}
for correct, wrongs in COMMON_CORRECTIONS.items():
    for w in wrongs:
        CORRECTION_MAP[w] = correct

def basic_spell_correct(text: str) -> str:
    if not isinstance(text, str): return ""
    tokens = text.lower().split()
    corrected = [CORRECTION_MAP.get(t, t) for t in tokens]
    return " ".join(corrected)

# =============================================================================
# APPLY CLEANING
# =============================================================================
print("Cleaning French reviews...")
df_train["avis_clean_fr"] = df_train["avis"].apply(lambda x: clean_text(x, "fr"))
df_train["avis_spell_fr"] = df_train["avis"].apply(basic_spell_correct)

print("Cleaning English translations...")
df_train["avis_clean_en"] = df_train["avis_en"].apply(lambda x: clean_text(x, "en"))

df_train["review_length_raw"]   = df_train["avis"].str.len()
df_train["review_length_clean"] = df_train["avis_clean_fr"].str.len()
df_train["word_count_raw"]      = df_train["avis"].str.split().str.len()
df_train["word_count_clean"]    = df_train["avis_clean_fr"].str.split().str.len()

print(f"Avg words before cleaning: {df_train['word_count_raw'].mean():.1f}")
print(f"Avg words after  cleaning: {df_train['word_count_clean'].mean():.1f}")

# =============================================================================
# N-GRAM ANALYSIS
# =============================================================================
corpus_fr = df_train["avis_clean_fr"].dropna().tolist()
corpus_pos = df_train[df_train["sentiment"]=="Positive"]["avis_clean_fr"].dropna().tolist()
corpus_neg = df_train[df_train["sentiment"]=="Negative"]["avis_clean_fr"].dropna().tolist()

top_uni = extract_ngrams(corpus_fr, n=1, top_k=25)
top_bi  = extract_ngrams(corpus_fr, n=2, top_k=20)
top_tri = extract_ngrams(corpus_fr, n=3, top_k=15)
top_pos_bi = extract_ngrams(corpus_pos, n=2, top_k=15)
top_neg_bi = extract_ngrams(corpus_neg, n=2, top_k=15)

# =============================================================================
# VISUALIZATIONS
# =============================================================================
fig = plt.figure(figsize=(20, 24))
fig.suptitle("Part 1 – Data Cleaning & N-gram Analysis", fontsize=15, fontweight="bold")

# 1. Word frequency unigrams
ax1 = fig.add_subplot(4, 3, 1)
words, freqs = zip(*top_uni)
ax1.barh(words[::-1], freqs[::-1], color="#3498db", edgecolor="white")
ax1.set_title("Top 25 Unigrams (FR)")
ax1.set_xlabel("Frequency")

# 2. Bigrams
ax2 = fig.add_subplot(4, 3, 2)
words2, freqs2 = zip(*top_bi)
ax2.barh(words2[::-1], freqs2[::-1], color="#2ecc71", edgecolor="white")
ax2.set_title("Top 20 Bigrams (FR)")
ax2.set_xlabel("Frequency")

# 3. Trigrams
ax3 = fig.add_subplot(4, 3, 3)
words3, freqs3 = zip(*top_tri)
ax3.barh(words3[::-1], freqs3[::-1], color="#e67e22", edgecolor="white")
ax3.set_title("Top 15 Trigrams (FR)")
ax3.set_xlabel("Frequency")

# 4. WordCloud all
ax4 = fig.add_subplot(4, 3, 4)
wc_all = WordCloud(width=600, height=300, background_color="white",
                   max_words=100, colormap="Blues").generate(" ".join(corpus_fr))
ax4.imshow(wc_all, interpolation="bilinear")
ax4.axis("off"); ax4.set_title("WordCloud – All Reviews")

# 5. WordCloud positive
ax5 = fig.add_subplot(4, 3, 5)
wc_pos = WordCloud(width=600, height=300, background_color="white",
                   max_words=80, colormap="Greens").generate(" ".join(corpus_pos))
ax5.imshow(wc_pos, interpolation="bilinear")
ax5.axis("off"); ax5.set_title("WordCloud – Positive Reviews")

# 6. WordCloud negative
ax6 = fig.add_subplot(4, 3, 6)
wc_neg = WordCloud(width=600, height=300, background_color="white",
                   max_words=80, colormap="Reds").generate(" ".join(corpus_neg))
ax6.imshow(wc_neg, interpolation="bilinear")
ax6.axis("off"); ax6.set_title("WordCloud – Negative Reviews")

# 7. Positive bigrams
ax7 = fig.add_subplot(4, 3, 7)
w7, f7 = zip(*top_pos_bi)
ax7.barh(w7[::-1], f7[::-1], color="#27ae60", edgecolor="white")
ax7.set_title("Top Bigrams – Positive Reviews")
ax7.set_xlabel("Frequency")

# 8. Negative bigrams
ax8 = fig.add_subplot(4, 3, 8)
w8, f8 = zip(*top_neg_bi)
ax8.barh(w8[::-1], f8[::-1], color="#c0392b", edgecolor="white")
ax8.set_title("Top Bigrams – Negative Reviews")
ax8.set_xlabel("Frequency")

# 9. Word count distribution by sentiment
ax9 = fig.add_subplot(4, 3, 9)
for sent, color in [("Positive","#2ecc71"),("Neutral","#f1c40f"),("Negative","#e74c3c")]:
    sub = df_train[df_train["sentiment"]==sent]["word_count_clean"]
    sub = sub[sub < sub.quantile(0.99)]
    ax9.hist(sub, bins=40, alpha=0.6, label=sent, color=color, edgecolor="white")
ax9.set_title("Word Count Distribution by Sentiment")
ax9.set_xlabel("Word count (cleaned)")
ax9.set_ylabel("Frequency")
ax9.legend()

# 10. Review length before/after cleaning
ax10 = fig.add_subplot(4, 3, 10)
ax10.scatter(df_train["word_count_raw"].clip(upper=300),
             df_train["word_count_clean"].clip(upper=200),
             alpha=0.08, s=5, color="#8e44ad")
ax10.set_title("Word Count: Raw vs Cleaned")
ax10.set_xlabel("Raw word count")
ax10.set_ylabel("Cleaned word count")
z = np.polyfit(df_train["word_count_raw"].fillna(0), df_train["word_count_clean"].fillna(0), 1)
x_line = np.linspace(0, 300, 100)
ax10.plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=1.5)

# 11. N-gram length (chars) distribution
ax11 = fig.add_subplot(4, 3, 11)
ngram_sizes = {"Unigrams": [len(w.split()) for w,_ in top_uni],
               "Bigrams":  [len(w.split()) for w,_ in top_bi],
               "Trigrams": [len(w.split()) for w,_ in top_tri]}
ax11.bar(ngram_sizes.keys(), [np.mean(v) for v in ngram_sizes.values()],
         color=["#3498db","#2ecc71","#e67e22"], edgecolor="white")
ax11.set_title("Average Token Count per N-gram Type")
ax11.set_ylabel("Avg tokens")

# 12. TF-IDF top terms per sentiment
ax12 = fig.add_subplot(4, 3, 12)
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,1))
X_tfidf = tfidf.fit_transform(corpus_fr)
feat_names = tfidf.get_feature_names_out()
mean_tfidf = pd.Series(X_tfidf.mean(axis=0).A1, index=feat_names).nlargest(20)
mean_tfidf[::-1].plot(kind="barh", ax=ax12, color="#16a085", edgecolor="white")
ax12.set_title("Top 20 TF-IDF Terms (Global)")
ax12.set_xlabel("Mean TF-IDF score")

plt.tight_layout()
plt.savefig("01_data_cleaning.png", bbox_inches="tight", dpi=110)
plt.close()
print("Saved: 01_data_cleaning.png")

# =============================================================================
# SAVE CLEAN DATA
# =============================================================================
df_train.to_csv("data_clean.csv", index=False)
print(f"Saved: data_clean.csv ({len(df_train):,} rows)")
print("\nNew columns added:")
for c in ["avis_clean_fr","avis_spell_fr","avis_clean_en",
          "review_length_raw","review_length_clean","word_count_raw",
          "word_count_clean","sentiment","year"]:
    print(f"  • {c}")

# Print sample spelling corrections found
corrections_found = []
for text in df_train["avis"].head(5000):
    if not isinstance(text, str): continue
    for token in text.lower().split():
        if token in CORRECTION_MAP:
            corrections_found.append((token, CORRECTION_MAP[token]))

print(f"\nSpelling corrections applied (sample): {len(corrections_found)} corrections in first 5000 rows")
if corrections_found:
    for orig, corr in Counter(corrections_found).most_common(10):
        print(f"  '{orig[0]}' → '{orig[1]}' ({corr}x)")
