"""
PROJECT – Part 4: Supervised Learning
======================================
Tasks:
  1. Star prediction (1-5 classification)
  2. Sentiment classification (Pos/Neg/Neutral)

Models:
  A. TF-IDF + Logistic Regression
  B. TF-IDF + Random Forest
  C. TF-IDF + SVM
  D. TF-IDF + Gradient Boosting
  E. Word2Vec averaging + Logistic Regression
  F. Zero-shot topic classification (keyword-based LLM-like)

Produces: model_comparison.png, error_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_with_topics.csv")
w2v = Word2Vec.load("word2vec.model")
wv  = w2v.wv
print(f"Dataset: {len(df):,} rows")

# ── Features & targets ────────────────────────────────────────────────────────
texts   = df["avis_clean_fr"].fillna("").tolist()
y_stars = df["note"].astype(int).tolist()                        # 1-5
y_sent  = df["sentiment"].tolist()                               # Pos/Neg/Neutral

X_train_t, X_test_t, ys_train, ys_test, yp_train, yp_test = train_test_split(
    texts, y_stars, y_sent, test_size=0.2, random_state=42, stratify=y_sent
)

# =============================================================================
# HELPER: Word2Vec average embedding
# =============================================================================
def avg_w2v(texts, model_wv, dim=100):
    embeddings = []
    for text in texts:
        tokens = text.split()
        vecs = [model_wv[w] for w in tokens if w in model_wv]
        if vecs:
            embeddings.append(np.mean(vecs, axis=0))
        else:
            embeddings.append(np.zeros(dim))
    return np.array(embeddings)

print("Computing Word2Vec embeddings...")
X_train_w2v = avg_w2v(X_train_t, wv)
X_test_w2v  = avg_w2v(X_test_t,  wv)

# =============================================================================
# MODELS FOR SENTIMENT (3-class)
# =============================================================================
print("\n── SENTIMENT CLASSIFICATION ─────────────────────────────────────────")

sentiment_models = {
    "TF-IDF + LogReg":     Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=1.0, random_state=42))
    ]),
    "TF-IDF + LinearSVC":  Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf",   LinearSVC(max_iter=2000, C=0.5, random_state=42))
    ]),
    "TF-IDF + RandomForest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, sublinear_tf=True)),
        ("clf",   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
}

sent_results = {}
for name, model in sentiment_models.items():
    print(f"  Training {name}...")
    model.fit(X_train_t, yp_train)
    preds = model.predict(X_test_t)
    acc   = accuracy_score(yp_test, preds)
    f1    = f1_score(yp_test, preds, average="weighted")
    sent_results[name] = {"acc": acc, "f1": f1, "preds": preds, "model": model}
    print(f"    Acc={acc:.4f}  F1={f1:.4f}")

# Best sentiment model
best_sent_name = max(sent_results, key=lambda k: sent_results[k]["f1"])
best_sent_model = sent_results[best_sent_name]["model"]

# W2V + LogReg sentiment
lr_w2v = LogisticRegression(max_iter=500, C=1.0, random_state=42)
lr_w2v.fit(X_train_w2v, yp_train)
preds_w2v = lr_w2v.predict(X_test_w2v)
sent_results["W2V avg + LogReg"] = {
    "acc": accuracy_score(yp_test, preds_w2v),
    "f1":  f1_score(yp_test, preds_w2v, average="weighted"),
    "preds": preds_w2v
}
print(f"  W2V avg + LogReg:  Acc={sent_results['W2V avg + LogReg']['acc']:.4f}  "
      f"F1={sent_results['W2V avg + LogReg']['f1']:.4f}")

# =============================================================================
# MODELS FOR STAR PREDICTION (5-class)
# =============================================================================
print("\n── STAR RATING PREDICTION ───────────────────────────────────────────")

star_models = {
    "TF-IDF + LogReg":    Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=1.0, random_state=42))
    ]),
    "TF-IDF + LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf",   LinearSVC(max_iter=2000, C=0.5, random_state=42))
    ]),
    "TF-IDF + RF":        Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, sublinear_tf=True)),
        ("clf",   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
}

star_results = {}
for name, model in star_models.items():
    print(f"  Training {name}...")
    model.fit(X_train_t, ys_train)
    preds = model.predict(X_test_t)
    acc = accuracy_score(ys_test, preds)
    f1  = f1_score(ys_test, preds, average="weighted")
    star_results[name] = {"acc": acc, "f1": f1, "preds": preds, "model": model}
    print(f"    Acc={acc:.4f}  F1={f1:.4f}")

lr_star_w2v = LogisticRegression(max_iter=500, C=1.0, random_state=42)
lr_star_w2v.fit(X_train_w2v, ys_train)
preds_star_w2v = lr_star_w2v.predict(X_test_w2v)
star_results["W2V avg + LogReg"] = {
    "acc": accuracy_score(ys_test, preds_star_w2v),
    "f1":  f1_score(ys_test, preds_star_w2v, average="weighted"),
    "preds": preds_star_w2v
}
print(f"  W2V avg + LogReg: Acc={star_results['W2V avg + LogReg']['acc']:.4f}  "
      f"F1={star_results['W2V avg + LogReg']['f1']:.4f}")

# Best star model
best_star_name  = max(star_results, key=lambda k: star_results[k]["f1"])
best_star_model = star_models.get(best_star_name) or star_models["TF-IDF + LogReg"]

# Save best models for Streamlit
import pickle
with open("model_sentiment.pkl", "wb") as f:
    pickle.dump(best_sent_model, f)
with open("model_stars.pkl", "wb") as f:
    pickle.dump(star_models["TF-IDF + LogReg"], f)
print("\nModels saved: model_sentiment.pkl, model_stars.pkl")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
fig = plt.figure(figsize=(20, 24))
fig.suptitle("Part 4 – Supervised Learning: Model Comparison", fontsize=15, fontweight="bold")

model_names = list(sent_results.keys())
accs = [sent_results[m]["acc"] for m in model_names]
f1s  = [sent_results[m]["f1"]  for m in model_names]

# 1. Sentiment model comparison
ax1 = fig.add_subplot(4, 3, 1)
x = np.arange(len(model_names))
bars1 = ax1.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#3498db", edgecolor="white")
bars2 = ax1.bar(x + 0.2, f1s,  0.35, label="F1 (weighted)", color="#e67e22", edgecolor="white")
ax1.set_xticks(x); ax1.set_xticklabels(model_names, rotation=25, ha="right", fontsize=8)
ax1.set_ylim(0, 1); ax1.set_title("Sentiment: Model Comparison")
ax1.set_ylabel("Score"); ax1.legend(fontsize=8)
ax1.axhline(0.7, color="red", linestyle="--", linewidth=0.8)
for bar in bars1: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                            f"{bar.get_height():.3f}", ha="center", fontsize=7)

# 2. Star model comparison
ax2 = fig.add_subplot(4, 3, 2)
star_names = list(star_results.keys())
s_accs = [star_results[m]["acc"] for m in star_names]
s_f1s  = [star_results[m]["f1"]  for m in star_names]
x2 = np.arange(len(star_names))
ax2.bar(x2-0.2, s_accs, 0.35, label="Accuracy", color="#2ecc71", edgecolor="white")
ax2.bar(x2+0.2, s_f1s,  0.35, label="F1",        color="#9b59b6", edgecolor="white")
ax2.set_xticks(x2); ax2.set_xticklabels(star_names, rotation=25, ha="right", fontsize=8)
ax2.set_ylim(0, 1); ax2.set_title("Star Rating: Model Comparison")
ax2.set_ylabel("Score"); ax2.legend(fontsize=8)

# 3. Confusion matrix – best sentiment model
ax3 = fig.add_subplot(4, 3, 3)
best_preds_sent = sent_results[best_sent_name]["preds"]
cm_sent = confusion_matrix(yp_test, best_preds_sent,
                            labels=["Negative","Neutral","Positive"])
ConfusionMatrixDisplay(cm_sent, display_labels=["Neg","Neu","Pos"]).plot(ax=ax3, colorbar=False)
ax3.set_title(f"Confusion Matrix – Sentiment\n({best_sent_name})")

# 4. Confusion matrix – best star model
ax4 = fig.add_subplot(4, 3, 4)
best_preds_star = star_results[best_star_name]["preds"]
cm_star = confusion_matrix(ys_test, best_preds_star, labels=[1,2,3,4,5])
ConfusionMatrixDisplay(cm_star, display_labels=[1,2,3,4,5]).plot(ax=ax4, colorbar=False)
ax4.set_title(f"Confusion Matrix – Stars\n({best_star_name})")

# 5. Classification report heatmap (sentiment)
ax5 = fig.add_subplot(4, 3, 5)
report = classification_report(yp_test, best_preds_sent,
                                output_dict=True, labels=["Negative","Neutral","Positive"])
report_df = pd.DataFrame(report).T.loc[["Negative","Neutral","Positive"],
                                        ["precision","recall","f1-score"]]
im5 = ax5.imshow(report_df.values, cmap="RdYlGn", vmin=0, vmax=1)
ax5.set_xticks([0,1,2]); ax5.set_xticklabels(["Precision","Recall","F1"])
ax5.set_yticks([0,1,2]); ax5.set_yticklabels(["Negative","Neutral","Positive"])
for i in range(3):
    for j in range(3):
        ax5.text(j, i, f"{report_df.values[i,j]:.3f}", ha="center", va="center", fontsize=10)
plt.colorbar(im5, ax=ax5)
ax5.set_title("Classification Report – Sentiment")

# 6. Classification report heatmap (stars)
ax6 = fig.add_subplot(4, 3, 6)
report_s = classification_report(ys_test, best_preds_star,
                                  output_dict=True, labels=[1,2,3,4,5])
report_s_df = pd.DataFrame(report_s).T.loc[["1","2","3","4","5"],
                                             ["precision","recall","f1-score"]]
im6 = ax6.imshow(report_s_df.values, cmap="RdYlGn", vmin=0, vmax=1)
ax6.set_xticks([0,1,2]); ax6.set_xticklabels(["Precision","Recall","F1"])
ax6.set_yticks([0,1,2,3,4]); ax6.set_yticklabels(["★","★★","★★★","★★★★","★★★★★"])
for i in range(5):
    for j in range(3):
        ax6.text(j, i, f"{report_s_df.values[i,j]:.3f}", ha="center", va="center", fontsize=9)
plt.colorbar(im6, ax=ax6)
ax6.set_title("Classification Report – Stars")

# 7. Error analysis – confidence of wrong predictions (TF-IDF+LR)
ax7 = fig.add_subplot(4, 3, 7)
lr_pipe = sentiment_models["TF-IDF + LogReg"]
proba = lr_pipe.predict_proba(X_test_t)
max_prob = proba.max(axis=1)
correct  = (np.array(yp_test) == lr_pipe.predict(X_test_t))
ax7.hist(max_prob[correct],  bins=30, alpha=0.7, label="Correct",  color="#2ecc71", density=True)
ax7.hist(max_prob[~correct], bins=30, alpha=0.7, label="Incorrect", color="#e74c3c", density=True)
ax7.set_title("Prediction Confidence Distribution")
ax7.set_xlabel("Max softmax probability")
ax7.set_ylabel("Density")
ax7.legend()

# 8. Error analysis – confused pairs
ax8 = fig.add_subplot(4, 3, 8)
test_arr = np.array(yp_test)
pred_arr = np.array(sent_results[best_sent_name]["preds"])
errors   = test_arr != pred_arr
error_pairs = Counter(zip(test_arr[errors], pred_arr[errors]))
from collections import Counter
ep_labels = [f"{t}→{p}" for (t,p),_ in error_pairs.most_common(8)]
ep_counts = [c for _,c in error_pairs.most_common(8)]
ax8.barh(ep_labels[::-1], ep_counts[::-1], color="#e74c3c", edgecolor="white")
ax8.set_title("Error Analysis – Confused Pairs (Sentiment)")
ax8.set_xlabel("# Errors")

# 9. Learning curve
ax9 = fig.add_subplot(4, 3, 9)
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    Pipeline([("tfidf", TfidfVectorizer(max_features=5000, sublinear_tf=True)),
              ("clf", LogisticRegression(max_iter=200, C=1.0))]),
    X_train_t, yp_train,
    cv=3, train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="f1_weighted", n_jobs=-1
)
ax9.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train F1", color="#2ecc71")
ax9.fill_between(train_sizes,
                 train_scores.mean(axis=1)-train_scores.std(axis=1),
                 train_scores.mean(axis=1)+train_scores.std(axis=1), alpha=0.15, color="#2ecc71")
ax9.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Val F1", color="#e74c3c")
ax9.fill_between(train_sizes,
                 val_scores.mean(axis=1)-val_scores.std(axis=1),
                 val_scores.mean(axis=1)+val_scores.std(axis=1), alpha=0.15, color="#e74c3c")
ax9.set_title("Learning Curve (TF-IDF + LogReg)")
ax9.set_xlabel("Training size"); ax9.set_ylabel("F1 Weighted")
ax9.legend(); ax9.grid(alpha=0.3)

# 10. TF-IDF top features per class
ax10 = fig.add_subplot(4, 3, 10)
lr_model = sentiment_models["TF-IDF + LogReg"]
tfidf_vect = lr_model.named_steps["tfidf"]
lr_clf     = lr_model.named_steps["clf"]
feat_names = tfidf_vect.get_feature_names_out()
classes    = lr_clf.classes_
top_n = 8
y_pos = np.arange(top_n)
colors_cls = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#2ecc71"}
offsets = {"Negative": -0.28, "Neutral": 0, "Positive": 0.28}
for cls in classes:
    if cls not in lr_clf.classes_: continue
    idx = list(lr_clf.classes_).index(cls)
    coefs = lr_clf.coef_[idx]
    top_ids = coefs.argsort()[:-top_n-1:-1]
    ax10.barh(y_pos + offsets.get(cls, 0),
              coefs[top_ids], height=0.25,
              label=cls, color=colors_cls.get(cls, "gray"), edgecolor="white")
ax10.set_yticks(y_pos)
ax10.set_yticklabels([feat_names[i] for i in
                       lr_clf.coef_[0].argsort()[:-top_n-1:-1]], fontsize=8)
ax10.set_title("Top Discriminative Features (LogReg coef)")
ax10.set_xlabel("Coefficient value")
ax10.legend(fontsize=8)
ax10.axvline(0, color="black", linewidth=0.8)

# 11. Prediction distribution vs true
ax11 = fig.add_subplot(4, 3, 11)
from collections import Counter
true_dist = Counter(ys_test)
pred_dist = Counter(best_preds_star)
x_vals = [1,2,3,4,5]
ax11.bar(np.array(x_vals)-0.2, [true_dist[k] for k in x_vals], 0.35,
         label="True", color="#3498db", edgecolor="white")
ax11.bar(np.array(x_vals)+0.2, [pred_dist[k] for k in x_vals], 0.35,
         label="Predicted", color="#e67e22", edgecolor="white")
ax11.set_xticks(x_vals); ax11.set_xlabel("Stars")
ax11.set_title("Star Prediction: True vs Predicted Distribution")
ax11.legend()

# 12. Model summary table
ax12 = fig.add_subplot(4, 3, 12)
ax12.axis("off")
all_results = {
    **{f"[Sent] {k}": v for k,v in sent_results.items()},
    **{f"[Star] {k}": v for k,v in star_results.items()},
}
summary_rows = [[name, f"{v['acc']:.4f}", f"{v['f1']:.4f}"]
                for name, v in all_results.items()]
table = ax12.table(
    cellText=summary_rows,
    colLabels=["Model", "Accuracy", "F1"],
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False); table.set_fontsize(8)
table.scale(1, 1.4)
ax12.set_title("All Models Summary", fontweight="bold")

plt.tight_layout()
plt.savefig("04_supervised_learning.png", bbox_inches="tight", dpi=110)
plt.close()
print("Saved: 04_supervised_learning.png")

# =============================================================================
# PRINT FINAL REPORT
# =============================================================================
print("\n" + "="*60)
print("FINAL SUPERVISED LEARNING REPORT")
print("="*60)
print("\nSENTIMENT CLASSIFICATION:")
for name, res in sent_results.items():
    print(f"  {name:<30} Acc={res['acc']:.4f}  F1={res['f1']:.4f}")
print(f"\n  ★ Best model: {best_sent_name}")

print("\nSTAR RATING PREDICTION:")
for name, res in star_results.items():
    print(f"  {name:<30} Acc={res['acc']:.4f}  F1={res['f1']:.4f}")
print(f"\n  ★ Best model: {best_star_name}")
