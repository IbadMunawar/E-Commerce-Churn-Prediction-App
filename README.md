<<<<<<< HEAD
# ChurnIQ — E-Commerce Customer Churn Intelligence Platform

An end-to-end ML web app that predicts customer churn and generates AI-powered business insights using Claude (Anthropic API).

## Features
- **EDA Dashboard** — Interactive charts, distributions, correlation heatmaps
- **ML Model Comparison** — Random Forest, Gradient Boosting, Logistic Regression with ROC curves
- **Single Customer Prediction** — Input customer details and get churn probability + AI retention strategy
- **AI Business Advisor** — Ask natural language questions about your churn data

## Tech Stack
`Python` · `Streamlit` · `Scikit-learn` · `Plotly` · `Anthropic Claude API`

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces
1. Create a new Space on huggingface.co (Streamlit SDK)
2. Upload `app.py`, `requirements.txt`, and `E_Commerce_Dataset.xlsx`
3. Add your Anthropic API key as a Space Secret named `ANTHROPIC_API_KEY`
4. Done — your app is live!

## Dataset
E-Commerce Customer Churn dataset with 5,630 customers and 20 features including Tenure, SatisfactionScore, Complaints, CashbackAmount, and more.

## Author
Ibad Munawar — Data Scientist | NED University
=======
# E-Commerce-Churn-Prediction-App
End-to-end ML web app for customer churn prediction with AI-powered insights
>>>>>>> 803c4b309a7c7632d4183423bd4d7e4b08ee588a
