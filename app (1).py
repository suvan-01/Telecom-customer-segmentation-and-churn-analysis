"""Telecom Customer Segmentation & Churn Analysis - Streamlit dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Telecom Churn Dashboard", page_icon="📡", layout="wide")

DATA_PATH = Path(__file__).parent / "IndiaTelecomData.csv"
USAGE = ["calls_made", "sms_sent", "data_used"]
NUMERIC = ["age", "num_dependents", "estimated_salary", "calls_made", "sms_sent", "data_used"]
CATEGORICAL = ["telecom_partner", "gender", "state", "city"]
SEG_FEATURES = NUMERIC
LABELS = {0: "Not churned", 1: "Churned"}


# ------------------------------------------------------------------ data
@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


@st.cache_data
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Negative usage -> NaN -> median imputation (same as the notebook)."""
    df = raw.copy()
    for c in USAGE:
        df.loc[df[c] < 0, c] = np.nan
        df[c] = df[c].fillna(df[c].median())
    return df


@st.cache_data
def segment(df: pd.DataFrame, k: int = 4):
    X = StandardScaler().fit_transform(df[SEG_FEATURES])
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    out = df.copy()
    out["cluster"] = labels
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    out["pc1"], out["pc2"] = coords[:, 0], coords[:, 1]
    return out


@st.cache_data
def k_search(df: pd.DataFrame):
    X = StandardScaler().fit_transform(df[SEG_FEATURES])
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        rows.append({"K": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def name_clusters(seg: pd.DataFrame) -> dict:
    """Auto-name each cluster from its most distinctive features."""
    prof = seg.groupby("cluster")[SEG_FEATURES].mean()
    z = (prof - prof.mean()) / prof.std(ddof=0)
    pretty = {"age": "age", "num_dependents": "dependents", "estimated_salary": "salary",
              "calls_made": "calling", "sms_sent": "SMS", "data_used": "data usage"}
    names = {}
    for c in z.index:
        top = z.loc[c].abs().sort_values(ascending=False).index[:2]
        parts = [("High " if z.loc[c, f] > 0 else "Low ") + pretty[f] for f in top]
        names[c] = f"Segment {c}: " + " / ".join(parts)
    return names


@st.cache_resource
def train_models(raw: pd.DataFrame):
    """Leak-free: invalid negatives -> NaN, imputed inside the pipeline (train split only)."""
    d = raw.copy()
    for c in USAGE:
        d.loc[d[c] < 0, c] = np.nan
    X = d[NUMERIC + CATEGORICAL]
    y = d["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    Xtr, Xte = pre.fit_transform(X_tr), pre.transform(X_te)
    specs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Balanced Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Balanced Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
        "Weighted Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight={0: 1, 1: 3}, n_jobs=-1),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
    }
    rows, cms, models = [], {}, {}
    for name, m in specs.items():
        m.fit(Xtr, y_tr)
        pred = m.predict(Xte)
        score = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred) * 100,
            "Precision": precision_score(y_te, pred, zero_division=0) * 100,
            "Recall": recall_score(y_te, pred, zero_division=0) * 100,
            "F1 Score": f1_score(y_te, pred, zero_division=0) * 100,
            "ROC-AUC": roc_auc_score(y_te, score),
        })
        cms[name] = confusion_matrix(y_te, pred)
        models[name] = m
    return pd.DataFrame(rows), cms, models, pre, y_te


def churn_rate_bar(df, col, title):
    g = df.groupby(col)["churn"].agg(["mean", "size"]).reset_index()
    g["Churn rate (%)"] = g["mean"] * 100
    fig = px.bar(g, x=col, y="Churn rate (%)", text=g["Churn rate (%)"].round(1),
                 hover_data={"size": True, "mean": False}, title=title)
    fig.add_hline(y=df["churn"].mean() * 100, line_dash="dash", annotation_text="overall")
    fig.update_traces(textposition="outside")
    return fig


def chi2_p(df, col):
    return chi2_contingency(pd.crosstab(df[col], df["churn"]))[1]


# ------------------------------------------------------------------ load
raw = load_raw()
df_all = clean(raw)
seg_all = segment(df_all)
cluster_names = name_clusters(seg_all)
seg_all["segment"] = seg_all["cluster"].map(cluster_names)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("📡 Telecom Churn")
page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Customer Segments",
                                     "Model Performance", "Try a Prediction", "Data Explorer"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
partners = st.sidebar.multiselect("Telecom partner", sorted(df_all.telecom_partner.unique()),
                                  default=sorted(df_all.telecom_partner.unique()))
genders = st.sidebar.multiselect("Gender", ["F", "M"], default=["F", "M"])
states = st.sidebar.multiselect("State (empty = all)", sorted(df_all.state.unique()))
age_rng = st.sidebar.slider("Age", int(df_all.age.min()), int(df_all.age.max()),
                            (int(df_all.age.min()), int(df_all.age.max())))
deps = st.sidebar.multiselect("Dependents", sorted(df_all.num_dependents.unique()),
                              default=sorted(df_all.num_dependents.unique()))
st.sidebar.caption("Filters apply to every page except Model Performance, "
                   "which always uses the fixed 20% test set.")

mask = (seg_all.telecom_partner.isin(partners) & seg_all.gender.isin(genders)
        & seg_all.age.between(*age_rng) & seg_all.num_dependents.isin(deps))
if states:
    mask &= seg_all.state.isin(states)
df = seg_all[mask]

st.title("Telecom Customer Segmentation & Churn Analysis")
st.caption("Indian telecom customer snapshot (2020). Findings describe this dataset only, "
           "not current telecom behaviour.")

if page not in ("Model Performance", "Try a Prediction") and df.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ------------------------------------------------------------------ pages
if page == "Overview":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churned", f"{int(df.churn.sum()):,}")
    c3.metric("Churn rate", f"{df.churn.mean()*100:.1f}%",
              delta=f"{(df.churn.mean()-df_all.churn.mean())*100:+.1f} pts vs all", delta_color="inverse")
    c4.metric("Avg. salary", f"₹{df.estimated_salary.mean():,.0f}")
    c5.metric("Avg. data used", f"{df.data_used.mean():,.0f}")

    l, r = st.columns(2)
    counts = df.churn.map(LABELS).value_counts().reset_index()
    counts.columns = ["Status", "Customers"]
    l.plotly_chart(px.pie(counts, names="Status", values="Customers", hole=0.5,
                          title="Churn distribution"), width="stretch")
    r.plotly_chart(churn_rate_bar(df, "telecom_partner", "Churn rate by telecom partner"),
                   width="stretch")

    l, r = st.columns(2)
    l.plotly_chart(churn_rate_bar(df, "gender", "Churn rate by gender"), width="stretch")
    r.plotly_chart(churn_rate_bar(df, "num_dependents", "Churn rate by number of dependents"),
                   width="stretch")

    st.subheader("Data quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows × columns", f"{raw.shape[0]:,} × {raw.shape[1]}")
    q2.metric("Duplicate rows", int(raw.duplicated().sum()))
    q3.metric("Invalid negative usage values fixed", int((raw[USAGE] < 0).sum().sum()))
    neg = (raw[USAGE] < 0).sum().rename("Negative values").reset_index()
    neg.columns = ["Column", "Negative values"]
    st.dataframe(neg, hide_index=True, width="content")

elif page == "EDA":
    st.subheader("Distributions by churn status")
    feat = st.selectbox("Feature", NUMERIC, index=3)
    plot_df = df.assign(Status=df.churn.map(LABELS))
    t1, t2 = st.columns(2)
    t1.plotly_chart(px.histogram(plot_df, x=feat, color="Status", barmode="overlay", opacity=0.6,
                                 nbins=30, title=f"{feat} by churn"), width="stretch")
    t2.plotly_chart(px.box(plot_df, x="Status", y=feat, color="Status",
                           title=f"{feat} box plot"), width="stretch")

    st.subheader("Mean behaviour: churned vs not churned")
    cmp_ = df.groupby(df.churn.map(LABELS))[NUMERIC].mean().round(2)
    st.dataframe(cmp_, width="stretch")

    l, r = st.columns(2)
    corr = df[NUMERIC + ["churn"]].corr().round(2)
    l.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Correlation matrix"), width="stretch")
    top_states = df.groupby("state")["churn"].agg(["mean", "size"]).reset_index()
    top_states = top_states[top_states["size"] >= 20].sort_values("mean", ascending=False).head(10)
    top_states["Churn rate (%)"] = top_states["mean"] * 100
    r.plotly_chart(px.bar(top_states, x="Churn rate (%)", y="state", orientation="h",
                          title="Top 10 states by churn rate (≥20 customers)")
                   .update_yaxes(autorange="reversed"), width="stretch")

    st.subheader("Is any difference statistically real?")
    tests = pd.DataFrame({
        "Variable": ["telecom_partner", "gender", "num_dependents", "city", "state"],
        "Chi-square p-value": [chi2_p(df, c) for c in
                               ["telecom_partner", "gender", "num_dependents", "city", "state"]],
    })
    tests["Significant at 5%?"] = np.where(tests["Chi-square p-value"] < 0.05, "Yes", "No")
    st.dataframe(tests.round(4), hide_index=True)
    st.caption("p-values above 0.05 mean the churn differences between groups are consistent with chance.")

elif page == "Customer Segments":
    st.subheader("K-Means segmentation (K = 4)")
    ks = k_search(df_all)
    l, r = st.columns(2)
    l.plotly_chart(px.line(ks, x="K", y="Inertia", markers=True, title="Elbow method"),
                   width="stretch")
    r.plotly_chart(px.line(ks, x="K", y="Silhouette", markers=True, title="Silhouette score"),
                   width="stretch")
    st.info(f"Silhouette at K=4 is {ks.loc[ks.K==4,'Silhouette'].iloc[0]:.3f}; values this low mean the "
            "segments are weakly separated and should be read as behavioural profiles, not hard groups.")

    prof = df.groupby("segment")[SEG_FEATURES].mean()
    size = df.groupby("segment").size().rename("Customers")
    churn = (df.groupby("segment")["churn"].mean() * 100).rename("Churn rate (%)")
    st.dataframe(pd.concat([size, prof.round(1), churn.round(2)], axis=1), width="stretch")

    l, r = st.columns(2)
    z = (prof - prof.mean()) / prof.std(ddof=0)
    l.plotly_chart(px.imshow(z.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                             aspect="auto", title="Segment profile (standardised means)"),
                   width="stretch")
    r.plotly_chart(px.bar(churn.reset_index(), x="segment", y="Churn rate (%)",
                          title="Churn rate by segment"), width="stretch")

    st.plotly_chart(px.scatter(df, x="pc1", y="pc2", color="segment", opacity=0.6,
                               title="Segments projected onto 2 principal components",
                               labels={"pc1": "PC1", "pc2": "PC2"}), width="stretch")
    p = chi2_p(df, "cluster") if df.cluster.nunique() > 1 else float("nan")
    st.caption(f"Chi-square test of churn vs segment: p = {p:.3f}. "
               "Segment churn rates are close, so segmentation does not separate churners.")

elif page == "Model Performance":
    results, cms, models, pre, y_te = train_models(raw)
    st.subheader("Model comparison (20% stratified test set, 406 customers)")
    base = y_te.mean() * 100
    best_auc = results["ROC-AUC"].max()
    c1, c2, c3 = st.columns(3)
    c1.metric("Majority-class accuracy", f"{100-base:.2f}%")
    c2.metric("Churn base rate in test set", f"{base:.1f}%")
    c3.metric("Best ROC-AUC", f"{best_auc:.3f}", help="0.5 = random guessing")
    fmt = {c: "{:.2f}" for c in ["Accuracy", "Precision", "Recall", "F1 Score"]}
    fmt["ROC-AUC"] = "{:.3f}"
    st.dataframe(results.style.format(fmt), hide_index=True, width="stretch")

    long = results.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1 Score"],
                        var_name="Metric", value_name="Score (%)")
    st.plotly_chart(px.bar(long, x="Model", y="Score (%)", color="Metric", barmode="group",
                           title="Precision, recall and F1 (churn class)"), width="stretch")
    st.warning("Accuracy is misleading here: predicting 'no churn' for everyone scores ~80%. "
               "ROC-AUC near 0.5 means none of the models ranks churners better than chance, "
               "and balanced-model precision is close to the base churn rate.")

    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for i, (name, cm) in enumerate(cms.items()):
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"],
                        title=name)
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10))
        cols[i % 3].plotly_chart(fig, width="stretch")

elif page == "Try a Prediction":
    results, cms, models, pre, y_te = train_models(raw)
    st.subheader("Score a customer")
    st.warning("Demo only: these models perform close to random on this dataset (see Model Performance).")
    model_name = st.selectbox("Model", ["Balanced Logistic Regression", "Balanced Random Forest"])
    a, b, c = st.columns(3)
    partner = a.selectbox("Telecom partner", sorted(df_all.telecom_partner.unique()))
    gender = a.selectbox("Gender", ["F", "M"])
    state = a.selectbox("State", sorted(df_all.state.unique()))
    city = b.selectbox("City", sorted(df_all.city.unique()))
    age = b.slider("Age", 18, 74, 40)
    deps_in = b.slider("Dependents", 0, 4, 2)
    salary = c.slider("Estimated salary", 20000, 150000, 85000, step=1000)
    calls = c.slider("Calls made", 0, 108, 49)
    sms = c.slider("SMS sent", 0, 53, 24)
    data = c.slider("Data used", 0, 10854, 5000, step=50)
    row = pd.DataFrame([{"telecom_partner": partner, "gender": gender, "state": state, "city": city,
                         "age": age, "num_dependents": deps_in, "estimated_salary": salary,
                         "calls_made": calls, "sms_sent": sms, "data_used": data}])
    prob = models[model_name].predict_proba(pre.transform(row))[0, 1]
    st.metric("Estimated churn probability", f"{prob*100:.1f}%",
              delta=f"{(prob - df_all.churn.mean())*100:+.1f} pts vs average", delta_color="inverse")
    st.progress(float(prob))

elif page == "Data Explorer":
    st.subheader(f"Customer records ({len(df):,})")
    only = st.radio("Show", ["All", "Churned only", "Not churned only"], horizontal=True)
    view = df if only == "All" else df[df.churn == (1 if only.startswith("Churned") else 0)]
    show_cols = [c for c in view.columns if c not in ("pc1", "pc2", "cluster")]
    st.dataframe(view[show_cols], hide_index=True, width="stretch")
    st.download_button("Download filtered CSV", view[show_cols].to_csv(index=False).encode("utf-8"),
                       "filtered_customers.csv", "text/csv")
    st.subheader("Summary statistics")
    st.dataframe(view[NUMERIC].describe().round(2), width="stretch")
