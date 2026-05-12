# ChurnIQ — E-Commerce Customer Churn Intelligence Platform

> End-to-end ML web app for customer churn prediction with AI-powered business insights

🚀 **Live Demo:** https://e-commerce-churn-prediction-app-gwnhq2x9bxrcovtstargmx.streamlit.app/

---

## 📌 Project Overview

This project analyzes an e-commerce dataset of 5,630 customers to understand and predict customer churn. It combines a **Machine Learning web app** built with Python/Streamlit and a **Power BI business intelligence dashboard** — giving both predictive and descriptive analytics in one project.

---

## 🤖 Machine Learning Web App (Streamlit)

### Features
- **EDA Dashboard** — Interactive charts, distributions, correlation heatmaps
- **ML Model Comparison** — Random Forest, Gradient Boosting, Logistic Regression with ROC curves
- **Single Customer Prediction** — Input customer details and get churn probability + AI retention strategy
- **AI Business Advisor** — Ask natural language questions about your churn data

### Tech Stack
`Python` · `Streamlit` · `Scikit-learn` · `Plotly` · `Gemini AI API`

### Model Performance
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Random Forest | 97.3% | 0.999 |
| Gradient Boosting | 93.1% | 0.944 |
| Logistic Regression | 85.2% | 0.857 |

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Power BI Dashboard

A two-page interactive business intelligence dashboard built in Power BI Desktop.

### What was done:
- **ETL in Power Query** — Imputed 1,000+ missing values with median, standardized messy categorical fields (merged "CC" → "Credit Card", "Phone" → "Mobile Phone")
- **DAX Measures** — Built dynamic KPIs: Churn Rate %, Total Customers, Avg Tenure, Avg Satisfaction
- **Data Binning** — Grouped CashbackAmount into $30 bins for trend analysis
- **Conditional Formatting** — Heatmap matrix showing churn by City Tier × Gender

### Key Business Insights Discovered:
- 📉 Churn spikes **50%+ in the first 2 months** — onboarding is the critical window
- 🏙️ **Tier 2 city females churn at 39%** — nearly double the overall average
- 💰 Cashback above **$160 cuts churn dramatically** — clear retention lever
- 😮 **Anomaly:** Customers giving a perfect "5" satisfaction score had the **highest churn rate (24%)** — high satisfaction does not equal loyalty

### Page 1 — Customer Retention & Churn Analytics
![Dashboard Page 1](images/dashboard_page1.jpg)

### Page 2 — Customer Behavior Deep Dive
![Dashboard Page 2](images/dashboard_page2.jpg)

---

## 📁 Dataset
E-Commerce Customer Churn dataset — 5,630 customers, 20 features including Tenure, SatisfactionScore, Complaints, CashbackAmount, PreferredOrderCategory, and more.

---

## 👤 Author
**Ibad Munawar** — Data Scientist | NED University of Engineering & Technology

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ibad%20Munawar-blue)](https://www.linkedin.com/in/ibad-munawar)
[![GitHub](https://img.shields.io/badge/GitHub-IbadMunawar-black)](https://github.com/IbadMunawar)
