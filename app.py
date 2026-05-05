import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.impute import SimpleImputer
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ – E-Commerce Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #666; font-size: 1rem; margin-top: 0; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07); text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #333; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .risk-high { color: #e74c3c; font-weight: 700; }
    .risk-low  { color: #27ae60; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .stButton>button:hover { opacity: 0.9; }
    .section-title {
        font-size: 1.3rem; font-weight: 600; color: #333;
        border-bottom: 2px solid #667eea; padding-bottom: 6px;
        margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def call_claude(prompt: str, max_tokens: int = 1000) -> str:
    import time
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=40
            )
            data = resp.json()
            # Rate limit — wait and retry
            if "error" in data and data["error"].get("code") == 429:
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                continue
            # Success
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    return candidate["content"]["parts"][0]["text"]
            if "promptFeedback" in data:
                return "⚠️ Request blocked by Gemini safety filters."
            return f"⚠️ Unexpected response: {str(data)[:200]}"
        except Exception as e:
            return f"⚠️ AI insight unavailable: {e}"
    return "⚠️ Rate limit hit — please wait 1 minute and try again."



@st.cache_data
def load_data(file) -> pd.DataFrame:
    if hasattr(file, "name") and file.name.endswith(".xlsx"):
        df = pd.read_excel(file, sheet_name=None)
        # pick the sheet that has a 'Churn' column
        for name, sheet in df.items():
            if "Churn" in sheet.columns:
                return sheet
        return list(df.values())[0]
    else:
        return pd.read_csv(file)


def preprocess(df: pd.DataFrame):
    df = df.copy()
    drop_cols = ["CustomerID"] if "CustomerID" in df.columns else []
    df.drop(columns=drop_cols, inplace=True)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "Churn"]

    # impute
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[num_cols] = num_imp.fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # encode
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = le.fit_transform(df[c].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y, cat_cols, num_cols


@st.cache_resource
def train_models(df_hash, _df):
    X, y, cat_cols, num_cols = preprocess(_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s  = sc.transform(X_test)

    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    results = {}
    for name, m in models.items():
        Xtr = X_train_s if name == "Logistic Regression" else X_train
        Xte = X_test_s  if name == "Logistic Regression" else X_test
        m.fit(Xtr, y_train)
        preds  = m.predict(Xte)
        probas = m.predict_proba(Xte)[:, 1]
        results[name] = {
            "model":    m,
            "accuracy": accuracy_score(y_test, preds),
            "auc":      roc_auc_score(y_test, probas),
            "report":   classification_report(y_test, preds, output_dict=True),
            "cm":       confusion_matrix(y_test, preds),
            "fpr_tpr":  roc_curve(y_test, probas),
            "X_test":   Xte,
            "y_test":   y_test,
        }

    best_name = max(results, key=lambda k: results[k]["auc"])
    fi = None
    best_m = results[best_name]["model"]
    if hasattr(best_m, "feature_importances_"):
        fi = pd.Series(best_m.feature_importances_, index=X.columns).sort_values(ascending=False)

    return results, best_name, fi, sc, X.columns.tolist()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 ChurnIQ")
    st.markdown("*E-Commerce Churn Intelligence*")
    st.divider()

    uploaded = st.file_uploader(
        "Upload your dataset",
        type=["xlsx", "csv"],
        help="Upload an Excel or CSV file with a 'Churn' column (0/1)"
    )
    use_default = st.checkbox("Use built-in sample dataset", value=True)
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📈 EDA & Insights",
        "🤖 ML Models",
        "🎯 Predict Customer",
        "💡 AI Business Advisor"
    ])

# ── Load data ─────────────────────────────────────────────────────────────────
df = None
if uploaded:
    df = load_data(uploaded)
    st.sidebar.success(f"✅ Loaded: {uploaded.name}")
elif use_default:
    try:
        df = pd.read_excel("E_Commerce_Dataset.xlsx", sheet_name="E Comm")
        st.sidebar.info("Using built-in dataset (5,630 customers)")
    except Exception as e:
        st.sidebar.error(f"Default dataset not found: {e}")

if df is None:
    st.markdown('<p class="main-header">📊 ChurnIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">E-Commerce Customer Churn Intelligence Platform</p>', unsafe_allow_html=True)
    st.info("👈 Upload your dataset or check 'Use built-in sample dataset' to get started.")
    st.stop()

# ── Train once ────────────────────────────────────────────────────────────────
results, best_name, feat_imp, scaler, feature_cols = train_models(
    id(df), df
)

churn_rate  = df["Churn"].mean() * 100
total_cust  = len(df)
churned     = df["Churn"].sum()
retained    = total_cust - churned

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">📊 ChurnIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">E-Commerce Customer Churn Intelligence Platform</p>', unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, f"{total_cust:,}", "Total Customers", "#667eea"),
        (c2, f"{churned:,}",   "Churned",          "#e74c3c"),
        (c3, f"{retained:,}", "Retained",          "#27ae60"),
        (c4, f"{churn_rate:.1f}%", "Churn Rate",   "#f39c12"),
    ]:
        col.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values=[retained, churned],
            names=["Retained", "Churned"],
            color_discrete_sequence=["#27ae60", "#e74c3c"],
            hole=0.55,
            title="Customer Retention Overview"
        )
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Best Model Performance</div>', unsafe_allow_html=True)
        for name, res in results.items():
            is_best = name == best_name
            st.markdown(f"""
            {'🏆 ' if is_best else '   '} **{name}**
            — Accuracy: `{res['accuracy']:.3f}` | AUC: `{res['auc']:.3f}`
            {'  ← **Best**' if is_best else ''}
            """)
        st.divider()
        st.markdown(f"**Dataset:** {total_cust:,} rows × {df.shape[1]} columns")
        st.markdown(f"**Missing values handled:** Yes (median/mode imputation)")
        st.markdown(f"**Train/Test split:** 80% / 20%")

    # Quick AI summary
    with st.expander("🤖 AI Executive Summary", expanded=True):
        if st.button("Generate AI Summary"):
            with st.spinner("Generating executive summary..."):
                prompt = f"""
You are a senior data scientist. Write a concise 4-sentence executive summary for a business stakeholder about this e-commerce churn dataset:
- Total customers: {total_cust:,}
- Churn rate: {churn_rate:.1f}%
- Best ML model: {best_name} with AUC {results[best_name]['auc']:.3f}
- Top churn drivers (from feature importance): {', '.join(feat_imp.head(5).index.tolist()) if feat_imp is not None else 'Tenure, Complaints, Cashback'}
Focus on business impact and what actions the company should take. Be direct and professional.
"""
                summary = call_claude(prompt, max_tokens=300)
            st.markdown(summary)
        else:
            st.info("Click the button above to generate an AI-powered executive summary.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA & Insights":
    st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribution", "Churn Breakdown", "Correlations"])

    with tab1:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Select a column to explore", num_cols)
        fig = px.histogram(df, x=col, color="Churn",
                           color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
                           barmode="overlay", nbins=40,
                           title=f"Distribution of {col} by Churn")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            cat = st.selectbox("Select categorical column", cat_cols)
            grp = df.groupby(cat)["Churn"].mean().reset_index()
            grp.columns = [cat, "Churn Rate"]
            grp["Churn Rate"] = grp["Churn Rate"] * 100
            grp = grp.sort_values("Churn Rate", ascending=False)
            fig = px.bar(grp, x=cat, y="Churn Rate",
                         color="Churn Rate",
                         color_continuous_scale="RdYlGn_r",
                         title=f"Churn Rate by {cat}")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_df = df.select_dtypes(include=np.number).fillna(df.median(numeric_only=True))
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    # AI insight on selected column
    st.divider()
    if st.button("🤖 Get AI Insight on This Data"):
        with st.spinner("Analyzing patterns..."):
            stats = df.groupby("Churn")[
                df.select_dtypes(include=np.number).columns[:8]
            ].mean().to_string()
            prompt = f"""
You are a data scientist. Analyze these mean feature values by churn status for an e-commerce company:

{stats}

Write 3 sharp, actionable insights (bullet points) that a business team can act on immediately.
Focus on which customer segments are most at risk and why. Be specific with numbers.
"""
            insight = call_claude(prompt, max_tokens=400)
        st.markdown(insight)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown('<div class="section-title">🤖 Machine Learning Model Comparison</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Model Comparison", "ROC Curves", "Feature Importance"])

    with tab1:
        rows = []
        for name, r in results.items():
            rep = r["report"]
            rows.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']:.4f}",
                "AUC-ROC":  f"{r['auc']:.4f}",
                "Precision (Churn)": f"{rep['1']['precision']:.4f}",
                "Recall (Churn)":    f"{rep['1']['recall']:.4f}",
                "F1 (Churn)":        f"{rep['1']['f1-score']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        st.success(f"🏆 Best model: **{best_name}** with AUC = {results[best_name]['auc']:.4f}")

        # Confusion matrix for best
        cm = results[best_name]["cm"]
        fig = px.imshow(cm, text_auto=True,
                        x=["Predicted: Retained", "Predicted: Churned"],
                        y=["Actual: Retained",    "Actual: Churned"],
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix — {best_name}")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        for name, r in results.items():
            fpr, tpr, _ = r["fpr_tpr"]
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['auc']:.3f})"
            ))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="gray"),
                                  name="Random Baseline"))
        fig.update_layout(
            title="ROC Curves – Model Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if feat_imp is not None:
            top = feat_imp.head(12).reset_index()
            top.columns = ["Feature", "Importance"]
            fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis",
                         title=f"Top 12 Feature Importances — {best_name}")
            fig.update_layout(yaxis=dict(autorange="reversed"),
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            if st.button("🤖 Explain Feature Importances with AI"):
                with st.spinner("Generating explanation..."):
                    top5 = feat_imp.head(5).to_dict()
                    prompt = f"""
You are a senior data scientist explaining model results to a business stakeholder.
The top 5 features driving customer churn in our e-commerce ML model are:
{top5}

For each feature, explain in plain English:
1. What it means
2. Why it drives churn
3. One concrete business action to address it

Be concise. Use bullet points. No technical jargon.
"""
                    explanation = call_claude(prompt, max_tokens=600)
                st.markdown(explanation)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT CUSTOMER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Customer":
    st.markdown('<div class="section-title">🎯 Predict Churn for a Single Customer</div>', unsafe_allow_html=True)
    st.info("Fill in the customer details below to get their churn probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 61, 10)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        warehouse_dist = st.slider("Warehouse to Home (km)", 5, 127, 20)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        complain = st.selectbox("Has Complained?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col2:
        hours_app = st.slider("Hours on App", 0, 5, 2)
        devices = st.slider("Devices Registered", 1, 6, 3)
        order_hike = st.slider("Order Amount Hike %", 11, 26, 15)
        coupons = st.slider("Coupons Used", 0, 16, 2)
        order_count = st.slider("Order Count", 1, 16, 3)

    with col3:
        days_since = st.slider("Days Since Last Order", 0, 46, 5)
        cashback = st.slider("Cashback Amount ($)", 0, 325, 150)
        login_device = st.selectbox("Preferred Login", ["Mobile Phone", "Computer", "Phone"])
        payment = st.selectbox("Payment Mode", ["Debit Card", "UPI", "Credit Card", "Cash on Delivery", "E wallet", "COD", "CC"])
        order_cat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"])

    if st.button("🔍 Predict Churn Risk"):
        # Build input — use median for gender/marital not shown
        sample = {
            "Tenure": tenure,
            "PreferredLoginDevice": login_device,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_dist,
            "PreferredPaymentMode": payment,
            "Gender": "Male",
            "HourSpendOnApp": hours_app,
            "NumberOfDeviceRegistered": devices,
            "PreferedOrderCat": order_cat,
            "SatisfactionScore": satisfaction,
            "MaritalStatus": "Single",
            "NumberOfAddress": 3,
            "Complain": complain,
            "OrderAmountHikeFromlastYear": order_hike,
            "CouponUsed": coupons,
            "OrderCount": order_count,
            "DaySinceLastOrder": days_since,
            "CashbackAmount": cashback,
        }
        sample_df = pd.DataFrame([sample])

        # preprocess same way
        X_full, y_full, _, _ = preprocess(df)
        le_map = {}
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for c in cat_cols:
            le = LabelEncoder()
            le.fit(df[c].astype(str))
            le_map[c] = le
            val = str(sample_df[c].iloc[0])
            if val in le.classes_:
                sample_df[c] = le.transform([val])
            else:
                sample_df[c] = 0

        num_imp = SimpleImputer(strategy="median")
        num_cols_feat = [c for c in feature_cols if c in sample_df.columns and c not in cat_cols]
        all_num = df[num_cols_feat].copy()
        num_imp.fit(all_num)
        sample_df[num_cols_feat] = num_imp.transform(sample_df[num_cols_feat])

        sample_df = sample_df[feature_cols]
        best_model = results[best_name]["model"]
        prob = best_model.predict_proba(sample_df)[0][1]
        pred = int(prob >= 0.5)

        st.divider()
        risk_color = "#e74c3c" if prob >= 0.5 else "#27ae60"
        risk_label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card" style="border-left-color:{risk_color}">
            <div class="metric-value" style="color:{risk_color}">{prob*100:.1f}%</div>
            <div class="metric-label">Churn Probability</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card" style="border-left-color:{risk_color}">
            <div class="metric-value" style="color:{risk_color}">{risk_label}</div>
            <div class="metric-label">Risk Classification</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best_name.split()[0]}</div>
            <div class="metric-label">Model Used</div></div>""", unsafe_allow_html=True)

        with st.spinner("Generating personalized retention strategy..."):
            prompt = f"""
A customer has a {prob*100:.1f}% churn probability in our e-commerce platform.

Customer profile:
- Tenure: {tenure} months
- Satisfaction Score: {satisfaction}/5
- Has Complained: {'Yes' if complain else 'No'}
- Days Since Last Order: {days_since}
- Order Count: {order_count}
- Cashback Amount: ${cashback}
- Preferred Category: {order_cat}

Write a specific 3-point retention action plan for this customer.
Be concrete: specify exact offers, timings, and communication channels.
Keep it under 150 words.
"""
            strategy = call_claude(prompt, max_tokens=300)
        st.markdown("### 💡 AI Retention Strategy")
        st.markdown(strategy)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI BUSINESS ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 AI Business Advisor":
    st.markdown('<div class="section-title">💡 AI Business Advisor</div>', unsafe_allow_html=True)
    st.markdown("Ask anything about your churn data and get AI-powered answers.")

    quick = st.selectbox("Quick questions", [
        "— Pick a question or type your own below —",
        "What are the top 3 reasons customers are churning?",
        "Which customer segment should we prioritize for retention?",
        "What retention campaigns would reduce churn the most?",
        "How does our churn rate compare to industry benchmarks?",
        "What is the estimated revenue impact of reducing churn by 5%?",
    ])

    user_q = st.text_area("Or ask your own question:", height=100,
                          placeholder="e.g. Why do mobile phone category customers churn more?")

    final_q = user_q.strip() if user_q.strip() else (quick if quick.startswith("What") or quick.startswith("Which") or quick.startswith("How") else "")

    if st.button("🤖 Ask AI Advisor") and final_q:
        # Build context from data
        churn_by_complaint = df.groupby("Complain")["Churn"].mean().to_dict()
        avg_tenure_churned = df[df["Churn"]==1]["Tenure"].mean()
        avg_tenure_retained = df[df["Churn"]==0]["Tenure"].mean()
        top_features = feat_imp.head(5).index.tolist() if feat_imp is not None else []

        context = f"""
Dataset context:
- Total customers: {total_cust:,}, Churn rate: {churn_rate:.1f}%
- Avg tenure churned: {avg_tenure_churned:.1f} months vs retained: {avg_tenure_retained:.1f} months
- Churn rate with complaint: {churn_by_complaint.get(1,0)*100:.1f}% vs without: {churn_by_complaint.get(0,0)*100:.1f}%
- Top predictive features: {', '.join(top_features)}
- Best ML model: {best_name} (AUC: {results[best_name]['auc']:.3f})
"""
        with st.spinner("AI Advisor is thinking..."):
            prompt = f"""
You are a senior business analytics consultant specializing in e-commerce customer retention.

{context}

User question: {final_q}

Provide a detailed, actionable answer with specific data-backed recommendations.
Use the dataset statistics above in your response. Be direct and business-focused.
Format with clear sections if needed. Max 300 words.
"""
            answer = call_claude(prompt, max_tokens=500)

        st.markdown(f"""<div class="insight-box">
        <strong>🤖 AI Advisor Response:</strong><br><br>{answer}
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Key Business Metrics at a Glance")
    c1, c2, c3 = st.columns(3)
    avg_cashback_churned  = df[df["Churn"]==1]["CashbackAmount"].mean()
    avg_cashback_retained = df[df["Churn"]==0]["CashbackAmount"].mean()
    complaint_churn_rate  = df[df["Complain"]==1]["Churn"].mean() * 100

    c1.metric("Avg Cashback – Churned",  f"${avg_cashback_churned:.0f}")
    c2.metric("Avg Cashback – Retained", f"${avg_cashback_retained:.0f}")
    c3.metric("Churn Rate w/ Complaint", f"{complaint_churn_rate:.1f}%")
