"""
Streamlit Application – Insurance Review NLP Platform
=======================================================
Features:
  1. Prediction: star rating + sentiment + topic
  2. Insurer Analysis: aggregated metrics + charts
  3. Review Search: keyword/filter search
  4. Explanation: LIME-style token importance
  5. RAG: retrieve similar reviews + generate summary
  6. QA: question answering over reviews
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Review NLP Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
  .metric-card {
    background: #f8f9fa; border-radius: 10px; padding: 16px;
    border-left: 4px solid #3498db; margin: 8px 0;
  }
  .positive { border-left-color: #2ecc71; }
  .negative { border-left-color: #e74c3c; }
  .neutral  { border-left-color: #f39c12; }
  .highlight-pos { background-color: #c8f7c5; padding: 2px 4px; border-radius: 3px; }
  .highlight-neg { background-color: #ffd7d7; padding: 2px 4px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Stopwords ─────────────────────────────────────────────────────────────────
FR_STOPS = {
    "le","la","les","de","du","des","un","une","et","en","au","aux","à","a",
    "ce","se","je","tu","il","elle","nous","vous","ils","elles","me","te",
    "lui","y","ne","pas","plus","par","sur","sous","dans","avec","pour",
    "que","qui","quoi","dont","où","ou","si","mais","donc","or","ni","car",
    "très","bien","tout","tous","toute","toutes","mon","ma","mes","ton","ta",
    "tes","son","sa","ses","notre","votre","leur","leurs","même","aussi",
    "avoir","être","faire","dire","aller","voir","vouloir","pouvoir",
    "ai","as","est","sont","ont","était","avait","été","fait","dit","va",
    "plus","peu","lors","alors","après","avant","comme","dont","encore",
    "toujours","jamais","déjà","autre","autres","ces","cet","cette","ceux",
    "ça","c","d","j","l","m","n","qu","s","t","y","là","ci",
}
EN_STOPS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","have","has","had","do","does","did",
    "not","no","i","you","he","she","it","we","they","this","that","also",
    "very","just","can","get","all","any","more","some","such","than","too",
}
STOPS = FR_STOPS | EN_STOPS

TOPIC_MAP = {
    0: "Customer Service", 1: "Claims Processing", 2: "Pricing & Contracts",
    3: "Coverage & Guarantees", 4: "Cancellation", 5: "Online / Digital",
    6: "Auto Insurance", 7: "Health Insurance",
}
TOPIC_KEYWORDS = {
    "Customer Service":    ["conseiller","service","telephone","appel","contact","accueil","equipe","reponse","aimable","ecoute"],
    "Claims Processing":   ["sinistre","remboursement","dossier","delai","indemnisation","declaration","prise","charge","expertise"],
    "Pricing & Contracts": ["prix","tarif","cotisation","prime","contrat","augmentation","cout","economique","cher","abordable"],
    "Coverage & Guarantees":["garantie","couverture","assurance","risque","protection","responsabilite","franchise"],
    "Cancellation":        ["resiliation","resilier","echeance","annulation","renouvellement","hamon","lettre"],
    "Online / Digital":    ["application","site","espace","ligne","interface","numerique","connexion","compte"],
    "Auto Insurance":      ["vehicule","voiture","accident","collision","conducteur","permis","route"],
    "Health Insurance":    ["sante","medecin","consultation","hospitalisation","optique","dentaire","mutuelle"],
}
POS_WORDS = {"satisfait","excellent","parfait","content","super","recommande","merci",
             "bravo","rapide","efficace","professionnel","agreable","serieux","top",
             "satisfied","excellent","perfect","happy","great","recommend","fast",
             "efficient","professional","pleasant","serious"}
NEG_WORDS = {"probleme","mauvais","terrible","nul","arnaque","honte","fuir","decevant",
             "insatisfait","lent","nul","refuses","impossible","incompetent","arnaque",
             "problem","bad","terrible","awful","scam","shame","flee","disappointing",
             "unsatisfied","slow","refuses","impossible","incompetent"}

# =============================================================================
# LOADERS (cached)
# =============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_with_topics.csv")
    except:
        df = pd.read_csv("avis_clients.csv")
        df = df[df["type"]=="train"].copy()
        df.drop(columns=["avis_cor","avis_cor_en"], inplace=True, errors="ignore")
        df.drop_duplicates(subset=["auteur","avis"], inplace=True)
        df["date_publication"] = pd.to_datetime(df["date_publication"], dayfirst=True)
        df["year"] = df["date_publication"].dt.year
        df["sentiment"] = df["note"].apply(
            lambda x: "Positive" if x>=4 else ("Neutral" if x==3 else "Negative"))
    return df

@st.cache_resource
def load_models():
    models = {}
    try:
        with open("model_sentiment.pkl","rb") as f: models["sentiment"] = pickle.load(f)
    except: models["sentiment"] = None
    try:
        with open("model_stars.pkl","rb") as f: models["stars"] = pickle.load(f)
    except: models["stars"] = None
    return models

@st.cache_resource
def build_tfidf_index(corpus):
    vec = TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)
    X   = vec.fit_transform(corpus)
    return vec, X

# =============================================================================
# HELPERS
# =============================================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"\b\d+([.,]\d+)?\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPS and len(t)>2]
    return " ".join(tokens)

def detect_topic(text):
    text_low = text.lower()
    scores = {}
    for topic, kws in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in kws if kw in text_low)
    best = max(scores, key=scores.get)
    return best, scores

def simple_sentiment(text):
    tokens = set(text.lower().split())
    pos_score = len(tokens & POS_WORDS)
    neg_score = len(tokens & NEG_WORDS)
    if pos_score > neg_score: return "Positive", pos_score / max(pos_score + neg_score, 1)
    if neg_score > pos_score: return "Negative", neg_score / max(pos_score + neg_score, 1)
    return "Neutral", 0.5

def rule_based_stars(text):
    text_low = text.lower()
    pos = sum(1 for w in POS_WORDS if w in text_low)
    neg = sum(1 for w in NEG_WORDS if w in text_low)
    score = pos - neg
    if score >= 3: return 5
    if score == 2: return 4
    if score == 0: return 3
    if score == -1: return 2
    return 1

def render_star(n):
    return "⭐" * int(n) + "☆" * (5 - int(n))

def highlight_tokens(text, pos_words, neg_words):
    tokens = text.split()
    result = []
    for tok in tokens:
        tok_clean = re.sub(r"[^\w]", "", tok.lower())
        if tok_clean in pos_words:
            result.append(f'<span class="highlight-pos">{tok}</span>')
        elif tok_clean in neg_words:
            result.append(f'<span class="highlight-neg">{tok}</span>')
        else:
            result.append(tok)
    return " ".join(result)

def semantic_search_reviews(query, vec, X, df, top_k=5):
    q_vec = vec.transform([clean_text(query)])
    sims  = cosine_similarity(q_vec, X).flatten()
    top_idx = sims.argsort()[:-top_k-1:-1]
    results = df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results

def extractive_summary(texts, n_sentences=3):
    """Extractive summary: pick most representative sentences by TF-IDF centrality."""
    all_sentences = []
    for t in texts:
        sents = re.split(r"[.!?]+", str(t))
        all_sentences.extend([s.strip() for s in sents if len(s.strip()) > 20])
    if not all_sentences: return "No text available."
    if len(all_sentences) <= n_sentences:
        return ". ".join(all_sentences) + "."
    try:
        tv = TfidfVectorizer(max_features=500)
        tv.fit(all_sentences)
        X_s = tv.transform(all_sentences)
        centroid = X_s.mean(axis=0)
        sims = cosine_similarity(centroid, X_s).flatten()
        top_ids = sims.argsort()[:-n_sentences-1:-1]
        top_ids_sorted = sorted(top_ids)
        return ". ".join([all_sentences[i] for i in top_ids_sorted]) + "."
    except:
        return ". ".join(all_sentences[:n_sentences]) + "."

def qa_over_reviews(question, reviews_text):
    """Simple keyword-based QA."""
    q_tokens = set(clean_text(question).split())
    sentences = []
    for t in reviews_text:
        sents = re.split(r"[.!?]+", str(t))
        sentences.extend([s.strip() for s in sents if len(s.strip()) > 15])
    scored = []
    for sent in sentences:
        s_tokens = set(clean_text(sent).split())
        overlap = len(q_tokens & s_tokens)
        if overlap > 0:
            scored.append((overlap, sent))
    scored.sort(reverse=True)
    if scored:
        return [s for _, s in scored[:3]]
    return ["No relevant answer found in the selected reviews."]

# =============================================================================
# MAIN APP
# =============================================================================
df = load_data()
models = load_models()

# Build TF-IDF index on a text column
text_col = "avis_clean_fr" if "avis_clean_fr" in df.columns else "avis"
corpus_for_index = df[text_col].fillna("").tolist()
tfidf_vec, tfidf_matrix = build_tfidf_index(corpus_for_index)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Insurance NLP Platform")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df):,} reviews")
st.sidebar.markdown(f"**Insurers:** {df['assureur'].nunique()}")
st.sidebar.markdown(f"**Products:** {df['produit'].nunique()}")
st.sidebar.markdown(f"**Avg Rating:** {df['note'].mean():.2f}/5")
st.sidebar.markdown("---")

tabs = st.tabs([
    "🔮 Prediction",
    "📊 Insurer Analysis",
    "🔍 Review Search",
    "💡 Explanation",
    "🔗 RAG",
    "❓ QA"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("🔮 Review Prediction")
    st.markdown("Enter an insurance review and get instant predictions.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_area(
            "Your review",
            placeholder="Ex: J'ai contacté le service client suite à mon accident, ils ont été très réactifs...",
            height=150,
            key="pred_input"
        )
        predict_btn = st.button("🚀 Predict", type="primary", key="predict_btn")
    
    with col2:
        st.markdown("**Examples:**")
        ex_texts = [
            "Service excellent, remboursement rapide et conseiller très aimable.",
            "Impossible de résilier, aucune réponse, je suis très déçu.",
            "Prix raisonnable mais la procédure de sinistre est longue.",
        ]
        for ex in ex_texts:
            if st.button(ex[:55]+"...", key=f"ex_{ex[:10]}"):
                user_input = ex
                predict_btn = True

    if predict_btn and user_input.strip():
        cleaned = clean_text(user_input)
        
        # Sentiment prediction
        if models["sentiment"]:
            try:
                sent_pred = models["sentiment"].predict([cleaned])[0]
                sent_proba = models["sentiment"].predict_proba([cleaned])[0]
                sent_conf  = max(sent_proba)
                sent_classes = models["sentiment"].classes_
            except:
                sent_pred, sent_conf = simple_sentiment(user_input)[:2]
                sent_proba = [0.33, 0.33, 0.33]
                sent_classes = ["Negative", "Neutral", "Positive"]
        else:
            sent_pred, sent_conf = simple_sentiment(user_input)[:2]
            sent_proba = [0.33, 0.33, 0.33]
            sent_classes = ["Negative", "Neutral", "Positive"]
        
        # Star prediction
        if models["stars"]:
            try:
                star_pred = models["stars"].predict([cleaned])[0]
                star_proba = models["stars"].predict_proba([cleaned])[0]
            except:
                star_pred = rule_based_stars(user_input)
                star_proba = None
        else:
            star_pred = rule_based_stars(user_input)
            star_proba = None
        
        # Topic
        topic_pred, topic_scores = detect_topic(user_input)
        
        st.markdown("---")
        st.subheader("📋 Results")
        
        c1, c2, c3 = st.columns(3)
        
        sent_color = {"Positive": "normal", "Negative": "inverse", "Neutral": "off"}
        with c1:
            st.metric("Sentiment", sent_pred,
                      f"Confidence: {sent_conf*100:.1f}%")
            if sent_pred == "Positive":
                st.success(f"😊 {sent_pred}")
            elif sent_pred == "Negative":
                st.error(f"😞 {sent_pred}")
            else:
                st.warning(f"😐 {sent_pred}")
        
        with c2:
            st.metric("Star Rating", render_star(star_pred), f"{star_pred}/5 stars")
            if star_proba is not None:
                fig_star, ax_star = plt.subplots(figsize=(4, 2))
                classes_str = [f"{'⭐'*i}" for i in (models["stars"].classes_
                               if models["stars"] else [1,2,3,4,5])]
                ax_star.bar(range(len(star_proba)), star_proba,
                            color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"],
                            edgecolor="white")
                ax_star.set_xticks(range(len(star_proba)))
                ax_star.set_xticklabels([str(i) for i in range(1,6)])
                ax_star.set_ylabel("Probability")
                ax_star.set_ylim(0,1)
                st.pyplot(fig_star, use_container_width=True)
                plt.close()
        
        with c3:
            st.metric("Topic", topic_pred)
            fig_topic, ax_t = plt.subplots(figsize=(4, 3))
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1])[-6:]
            ax_t.barh([t for t,_ in sorted_topics], [s for _,s in sorted_topics],
                     color="#3498db", edgecolor="white")
            ax_t.set_xlabel("Keyword matches")
            ax_t.tick_params(axis="y", labelsize=7)
            st.pyplot(fig_topic, use_container_width=True)
            plt.close()
        
        # Probability breakdown for sentiment
        if len(sent_proba) == len(sent_classes):
            st.subheader("Sentiment Probability Breakdown")
            fig_sent, ax_s = plt.subplots(figsize=(6, 2))
            colors_s = [{"Positive":"#2ecc71","Negative":"#e74c3c","Neutral":"#f39c12"}.get(c,"#bdc3c7")
                        for c in sent_classes]
            ax_s.barh(sent_classes, sent_proba, color=colors_s, edgecolor="white")
            ax_s.set_xlim(0, 1)
            ax_s.set_xlabel("Probability")
            for i, (cls, prob) in enumerate(zip(sent_classes, sent_proba)):
                ax_s.text(prob + 0.01, i, f"{prob*100:.1f}%", va="center")
            st.pyplot(fig_sent, use_container_width=True)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: INSURER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("📊 Insurer Analysis")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_reviews = st.slider("Min reviews per insurer", 10, 200, 50)
    with col_f2:
        selected_product = st.selectbox("Filter by product", ["All"] + sorted(df["produit"].unique()))
    
    df_filt = df.copy()
    if selected_product != "All":
        df_filt = df_filt[df_filt["produit"] == selected_product]
    
    ins_stats = (df_filt.groupby("assureur")
                 .agg(avg_note=("note","mean"),
                      n=("note","count"),
                      pct_pos=("sentiment", lambda x: (x=="Positive").mean()*100),
                      pct_neg=("sentiment", lambda x: (x=="Negative").mean()*100))
                 .query(f"n >= {min_reviews}")
                 .sort_values("avg_note", ascending=False))
    
    st.markdown(f"**{len(ins_stats)} insurers** with ≥{min_reviews} reviews")
    
    tab_a, tab_b, tab_c = st.tabs(["Rankings", "Charts", "Summary"])
    
    with tab_a:
        st.dataframe(ins_stats.reset_index().style.format({
            "avg_note": "{:.2f}", "n": "{:,}", "pct_pos": "{:.1f}%", "pct_neg": "{:.1f}%"
        }).background_gradient(subset=["avg_note"], cmap="RdYlGn"))
    
    with tab_b:
        # Top 15 / Bottom 15
        c1, c2 = st.columns(2)
        with c1:
            top15 = ins_stats.head(15)
            fig_top = px.bar(top15.reset_index(), x="avg_note", y="assureur",
                             orientation="h", color="avg_note",
                             color_continuous_scale="RdYlGn",
                             range_color=[1,5],
                             title="Top 15 Insurers",
                             labels={"avg_note": "Avg Rating", "assureur": ""},
                             text="avg_note")
            fig_top.update_traces(texttemplate="%{text:.2f}")
            fig_top.update_layout(height=450, showlegend=False,
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)
        
        with c2:
            bot15 = ins_stats.tail(15).sort_values("avg_note")
            fig_bot = px.bar(bot15.reset_index(), x="avg_note", y="assureur",
                             orientation="h", color="avg_note",
                             color_continuous_scale="RdYlGn",
                             range_color=[1,5],
                             title="Bottom 15 Insurers",
                             labels={"avg_note": "Avg Rating", "assureur": ""},
                             text="avg_note")
            fig_bot.update_traces(texttemplate="%{text:.2f}")
            fig_bot.update_layout(height=450, showlegend=False,
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_bot, use_container_width=True)
        
        # Volume vs rating scatter
        fig_scatter = px.scatter(
            ins_stats.reset_index(), x="n", y="avg_note",
            size="n", color="avg_note",
            color_continuous_scale="RdYlGn", range_color=[1,5],
            hover_name="assureur", log_x=True,
            title="Volume vs Average Rating per Insurer",
            labels={"n": "# Reviews (log)", "avg_note": "Avg Rating"}
        )
        fig_scatter.add_hline(y=df["note"].mean(), line_dash="dash",
                               line_color="navy", annotation_text="Global mean")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab_c:
        # Insurer detail
        selected_ins = st.selectbox("Select an insurer", ins_stats.index.tolist())
        ins_df = df_filt[df_filt["assureur"] == selected_ins]
        
        ci1, ci2, ci3, ci4 = st.columns(4)
        ci1.metric("Reviews", f"{len(ins_df):,}")
        ci2.metric("Avg Rating", f"{ins_df['note'].mean():.2f}/5")
        ci3.metric("% Positive", f"{(ins_df['sentiment']=='Positive').mean()*100:.1f}%")
        ci4.metric("% Negative", f"{(ins_df['sentiment']=='Negative').mean()*100:.1f}%")
        
        # Extractive summary
        st.subheader(f"📝 Summary for {selected_ins}")
        pos_reviews = ins_df[ins_df["sentiment"]=="Positive"]["avis"].dropna().tolist()
        neg_reviews = ins_df[ins_df["sentiment"]=="Negative"]["avis"].dropna().tolist()
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.success("**Positive highlights:**")
            st.write(extractive_summary(pos_reviews[:100], n_sentences=3))
        with sc2:
            st.error("**Negative highlights:**")
            st.write(extractive_summary(neg_reviews[:100], n_sentences=3))
        
        # Topic breakdown for this insurer
        if "lda_topic_label" in ins_df.columns:
            topic_dist = ins_df.groupby("lda_topic_label")["note"].agg(["mean","count"])
            topic_dist.columns = ["avg_note","count"]
            fig_t = px.bar(topic_dist.reset_index(), x="lda_topic_label", y="avg_note",
                           color="avg_note", color_continuous_scale="RdYlGn",
                           range_color=[1,5], text="avg_note",
                           title=f"Avg Rating by Topic – {selected_ins}",
                           labels={"lda_topic_label":"Topic","avg_note":"Avg Rating"})
            fig_t.update_traces(texttemplate="%{text:.2f}")
            st.plotly_chart(fig_t, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: REVIEW SEARCH (Information Retrieval)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🔍 Review Search")
    
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        search_query = st.text_input("Keyword search", placeholder="remboursement sinistre...")
    with sc2:
        filter_insurer = st.selectbox("Insurer", ["All"] + sorted(df["assureur"].unique()))
    with sc3:
        filter_product = st.selectbox("Product", ["All"] + sorted(df["produit"].unique()))
    with sc4:
        filter_stars = st.multiselect("Stars", [1,2,3,4,5], default=[1,2,3,4,5])
    
    search_btn = st.button("🔍 Search", type="primary")
    
    if search_btn or search_query:
        results_df = df.copy()
        
        if filter_insurer != "All":
            results_df = results_df[results_df["assureur"] == filter_insurer]
        if filter_product != "All":
            results_df = results_df[results_df["produit"] == filter_product]
        if filter_stars:
            results_df = results_df[results_df["note"].isin(filter_stars)]
        
        if search_query.strip():
            # Semantic TF-IDF search on filtered corpus
            sub_corpus = results_df[text_col].fillna("").tolist()
            if sub_corpus:
                sub_vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
                try:
                    sub_X = sub_vec.fit_transform(sub_corpus)
                    q_vec = sub_vec.transform([clean_text(search_query)])
                    sims  = cosine_similarity(q_vec, sub_X).flatten()
                    results_df = results_df.copy()
                    results_df["similarity"] = sims
                    results_df = results_df[results_df["similarity"] > 0.01].sort_values("similarity", ascending=False)
                except:
                    results_df["similarity"] = 0.0
        
        st.info(f"Found **{len(results_df):,}** reviews matching your criteria.")
        
        if len(results_df) > 0:
            display_df = results_df[["note","assureur","produit","avis","date_publication"]].head(50)
            display_df = display_df.rename(columns={"note":"⭐","assureur":"Insurer",
                                                     "produit":"Product","avis":"Review",
                                                     "date_publication":"Date"})
            display_df["Review"] = display_df["Review"].str[:200] + "..."
            st.dataframe(display_df, height=400)
            
            # Distribution of results
            c1, c2 = st.columns(2)
            with c1:
                fig_r = px.histogram(results_df, x="note", nbins=5,
                                     color_discrete_sequence=["#3498db"],
                                     title="Rating Distribution in Results",
                                     labels={"note":"Stars"})
                st.plotly_chart(fig_r, use_container_width=True)
            with c2:
                top_ins = results_df["assureur"].value_counts().head(10)
                fig_i = px.bar(x=top_ins.index, y=top_ins.values,
                               title="Top Insurers in Results",
                               labels={"x":"Insurer","y":"Count"})
                st.plotly_chart(fig_i, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: EXPLANATION (LIME-style)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("💡 Prediction Explanation")
    st.markdown("Understand **why** the model made its prediction — token-level importance.")
    
    exp_input = st.text_area("Review to explain", height=120,
                              placeholder="Paste any review here...",
                              key="exp_input")
    explain_btn = st.button("💡 Explain", type="primary")
    
    if explain_btn and exp_input.strip():
        st.subheader("Token Importance (positive = green, negative = red)")
        
        # Highlight known sentiment words
        highlighted = highlight_tokens(exp_input, POS_WORDS, NEG_WORDS)
        st.markdown(highlighted, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Show token-level scores
        tokens = exp_input.split()
        token_scores = []
        for tok in tokens:
            tok_clean = re.sub(r"[^\w]","", tok.lower())
            if tok_clean in POS_WORDS:   token_scores.append((tok, +1.0))
            elif tok_clean in NEG_WORDS: token_scores.append((tok, -1.0))
            elif tok_clean in STOPS:     token_scores.append((tok, 0.0))
            else:                        token_scores.append((tok, 0.1))
        
        meaningful = [(t, s) for t, s in token_scores if abs(s) > 0.05]
        meaningful.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if meaningful:
            fig_exp, ax_exp = plt.subplots(figsize=(8, max(3, len(meaningful[:15])*0.4)))
            words_exp  = [t for t,_ in meaningful[:15]]
            scores_exp = [s for _,s in meaningful[:15]]
            colors_exp = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores_exp]
            ax_exp.barh(words_exp[::-1], scores_exp[::-1], color=colors_exp[::-1],
                        edgecolor="white")
            ax_exp.axvline(0, color="black", linewidth=0.8)
            ax_exp.set_title("Token Contribution to Sentiment")
            ax_exp.set_xlabel("Sentiment score")
            st.pyplot(fig_exp, use_container_width=True)
            plt.close()
        
        # Final prediction
        topic, _ = detect_topic(exp_input)
        sentiment, conf = simple_sentiment(exp_input)
        stars = rule_based_stars(exp_input)
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Sentiment", sentiment, f"{conf*100:.0f}% confidence")
        c2.metric("Predicted Stars", render_star(stars))
        c3.metric("Detected Topic", topic)
        
        # TF-IDF feature importance if model available
        if models["sentiment"] and hasattr(models["sentiment"], "named_steps"):
            try:
                tfidf_step = models["sentiment"].named_steps["tfidf"]
                lr_step    = models["sentiment"].named_steps["clf"]
                vec_query  = tfidf_step.transform([clean_text(exp_input)])
                feat_names = tfidf_step.get_feature_names_out()
                
                pred_class = models["sentiment"].predict([clean_text(exp_input)])[0]
                cls_idx = list(lr_step.classes_).index(pred_class)
                coefs = lr_step.coef_[cls_idx]
                
                # Which features are present AND have high coefs
                present = vec_query.nonzero()[1]
                feature_importance = [(feat_names[i], float(vec_query[0, i]) * coefs[i])
                                       for i in present]
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                if feature_importance:
                    st.subheader(f"Model Feature Importance for '{pred_class}'")
                    fi_words  = [f for f,_ in feature_importance[:12]]
                    fi_scores = [s for _,s in feature_importance[:12]]
                    fi_colors = ["#2ecc71" if s>0 else "#e74c3c" for s in fi_scores]
                    fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
                    ax_fi.barh(fi_words[::-1], fi_scores[::-1], color=fi_colors[::-1],
                               edgecolor="white")
                    ax_fi.axvline(0, color="black", linewidth=0.8)
                    ax_fi.set_title(f"TF-IDF × LogReg coef (class='{pred_class}')")
                    ax_fi.set_xlabel("TF-IDF weight × coef")
                    st.pyplot(fig_fi, use_container_width=True)
                    plt.close()
            except Exception as e:
                st.info(f"Model-based explanation not available: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: RAG (Retrieve + Summarize)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🔗 RAG – Retrieval Augmented Generation")
    st.markdown("Find the most relevant reviews and generate an extractive summary.")
    
    rag_query = st.text_input("What are you looking for?",
                               placeholder="e.g. What do customers say about claims processing speed?")
    
    rag_col1, rag_col2 = st.columns(2)
    with rag_col1:
        rag_insurer = st.selectbox("Restrict to insurer (optional)", ["All"] + sorted(df["assureur"].unique()))
    with rag_col2:
        rag_topk = st.slider("Number of reviews to retrieve", 5, 30, 10)
    
    rag_btn = st.button("🔗 Retrieve & Summarize", type="primary")
    
    if rag_btn and rag_query.strip():
        rag_df = df.copy()
        if rag_insurer != "All":
            rag_df = rag_df[rag_df["assureur"] == rag_insurer]
        
        rag_corpus = rag_df[text_col].fillna("").tolist()
        
        # Retrieve
        with st.spinner("Searching relevant reviews..."):
            try:
                rv = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
                rX = rv.fit_transform(rag_corpus)
                q_vec = rv.transform([clean_text(rag_query)])
                sims = cosine_similarity(q_vec, rX).flatten()
                top_ids = sims.argsort()[:-rag_topk-1:-1]
                retrieved = rag_df.iloc[top_ids].copy()
                retrieved["similarity"] = sims[top_ids]
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                retrieved = rag_df.head(rag_topk).copy()
                retrieved["similarity"] = 0.0
        
        st.success(f"✅ Retrieved {len(retrieved)} relevant reviews")
        
        # Summary stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Rating (retrieved)", f"{retrieved['note'].mean():.2f}/5")
        c2.metric("% Positive", f"{(retrieved['sentiment']=='Positive').mean()*100:.0f}%")
        c3.metric("% Negative", f"{(retrieved['sentiment']=='Negative').mean()*100:.0f}%")
        
        # Extractive summary
        st.subheader("📝 Generated Summary")
        raw_texts = retrieved["avis"].dropna().tolist()
        summary = extractive_summary(raw_texts, n_sentences=4)
        st.info(summary)
        
        # Show retrieved reviews
        with st.expander("📋 View retrieved reviews"):
            for _, row in retrieved.iterrows():
                sentiment_icon = "✅" if row.get("sentiment") == "Positive" else ("❌" if row.get("sentiment") == "Negative" else "⚪")
                st.markdown(f"""
**{row['assureur']}** | {render_star(int(row['note']))} | {sentiment_icon} {row.get('sentiment','')}
*Similarity: {row['similarity']:.3f}*

> {str(row['avis'])[:300]}...

---""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: QA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("❓ Question Answering")
    st.markdown("Ask a specific question and get answers extracted from real reviews.")
    
    qa_insurer = st.selectbox("Select insurer (or All)", ["All"] + sorted(df["assureur"].unique()), key="qa_ins")
    qa_product = st.selectbox("Select product (optional)", ["All"] + sorted(df["produit"].unique()), key="qa_prod")
    
    qa_df = df.copy()
    if qa_insurer != "All": qa_df = qa_df[qa_df["assureur"] == qa_insurer]
    if qa_product != "All": qa_df = qa_df[qa_df["produit"] == qa_product]
    
    st.info(f"Knowledge base: **{len(qa_df):,} reviews** selected")
    
    # Suggested questions
    st.markdown("**Suggested questions:**")
    sugg_cols = st.columns(3)
    suggested = [
        "How fast is the reimbursement?",
        "Is the customer service good?",
        "Are there cancellation issues?",
        "What do people say about pricing?",
        "Are claims processed quickly?",
        "What are common complaints?",
    ]
    for i, q in enumerate(suggested):
        with sugg_cols[i % 3]:
            if st.button(q, key=f"sugg_{i}"):
                st.session_state["qa_question"] = q
    
    qa_question = st.text_input("Your question:", 
                                  value=st.session_state.get("qa_question", ""),
                                  placeholder="e.g. How long does it take to get reimbursed?",
                                  key="qa_q_input")
    qa_btn = st.button("❓ Find Answer", type="primary")
    
    if qa_btn and qa_question.strip():
        with st.spinner("Searching for relevant passages..."):
            reviews_text = qa_df["avis"].dropna().tolist()
            answers = qa_over_reviews(qa_question, reviews_text[:3000])
        
        st.subheader("📌 Most Relevant Passages")
        for i, ans in enumerate(answers, 1):
            st.markdown(f"**Answer {i}:**")
            st.success(ans)
        
        # Stats related to the question
        qa_topic, _ = detect_topic(qa_question)
        st.markdown(f"---\n*Topic detected in question: **{qa_topic}***")
        
        if "lda_topic_label" in qa_df.columns:
            related = qa_df[qa_df["lda_topic_label"] == qa_topic]
            if len(related) > 0:
                st.metric(f"Reviews about '{qa_topic}'", len(related),
                          f"Avg rating: {related['note'].mean():.2f}/5")
                
                # Sample reviews for this topic
                with st.expander(f"Sample reviews about '{qa_topic}'"):
                    for _, row in related.sample(min(5, len(related)), random_state=42).iterrows():
                        st.markdown(f"**{render_star(int(row['note']))}** | *{row['assureur']}*")
                        st.write(str(row['avis'])[:250] + "...")
                        st.divider()

# Footer
st.markdown("---")
st.markdown("*Insurance Review NLP Platform — Built with Streamlit, scikit-learn, Gensim*")
