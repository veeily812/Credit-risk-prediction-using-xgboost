# 🏦 Credit Risk Predictor

This Space is a **Gradio-powered web app** that uses a machine learning model (XGBoost) to predict **credit risk** based on customer financial data. It helps financial institutions assess whether an applicant is likely to default on a loan.

## 📊 Features
- Accepts user inputs like age, income, job type, credit score, loan amount, and default history
- Returns a risk prediction: **Low Risk** or **High Risk**
- Shows confidence scores from the model
- Can be extended with SHAP explainability

## 🧠 Model
- Trained on the **German Credit Dataset**
- Model: `XGBoostClassifier`
- Preprocessing includes scaling with `StandardScaler`

## 📂 Files
- `app.py`: Gradio UI
- `model.pkl`: Trained XGBoost model
- `scaler.pkl`: Pre-fitted scaler
- `requirements.txt`: Dependencies for Hugging Face
- `explainer.py`: (Optional) SHAP integration

## 🚀 Usage
Click the **"Open in Spaces"** button to interact with the app live!

## 🛠️ Future Enhancements
- SHAP explainability plots
- CSV batch prediction
- Downloadable PDF risk reports
- Dashboard view of flagged entries

---

### 🙋‍♂️ Author
PhunVi – Made with ❤️ using Hugging Face Spaces + Gradio
liveat:
https://huggingface.co/spaces/PhunvVi/Credit-risk-predictor
