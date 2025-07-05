import gradio as gr
import pickle
import numpy as np
import pandas as pd
from explainer import get_shap_values
import shap
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
from huggingface_hub import HfApi

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# List of columns used during training
columns = ['Age', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration',
    'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education', 'Purpose_furniture/equipment',
    'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others', 'Sex_female', 'Sex_male',
    'Housing_free', 'Housing_own', 'Housing_rent', 'Job_0', 'Job_1', 'Job_2', 'Job_3']

# Define prediction function
# All categorical mappings must match the encoding used during training
saving_account_map = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3, 'No Savings': 0}
checking_account_map = {'little': 0, 'moderate': 1, 'rich': 2, 'No Checking': 0}

purpose_options = ['business', 'car', 'domestic appliances', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others']
sex_options = ['female', 'male']
housing_options = ['free', 'own', 'rent']
job_options = [0, 1, 2, 3]

def to_scalar(val):
    # If val is a numpy array, get the first element; else, return as float
    if isinstance(val, np.ndarray):
        return float(val.flatten()[0])
    return float(val)

def predict_and_explain(age, saving_account, checking_account, credit_amount, duration, purpose, sex, housing, job):
    # Build a single-row DataFrame with all columns set to 0
    input_data = pd.DataFrame([[0]*len(columns)], columns=columns)
    input_data['Age'] = age
    input_data['Saving accounts'] = saving_account_map.get(saving_account, 0)
    input_data['Checking account'] = checking_account_map.get(checking_account, 0)
    input_data['Credit amount'] = credit_amount
    input_data['Duration'] = duration

    # One-hot encoding for purpose
    for p in purpose_options:
        input_data[f'Purpose_{p}'] = 1 if purpose == p else 0
    # One-hot encoding for sex
    for s in sex_options:
        input_data[f'Sex_{s}'] = 1 if sex == s else 0
    # One-hot encoding for housing
    for h in housing_options:
        input_data[f'Housing_{h}'] = 1 if housing == h else 0
    # One-hot encoding for job
    for j in job_options:
        input_data[f'Job_{j}'] = 1 if job == j else 0
 
    # Predict
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][int(pred)]
    risk = "Low Risk" if pred == 0 else "High Risk"
    prediction_text = f"{risk} (Confidence: {prob:.2f})"

    # Save flagged (high risk) input
    if risk == "High Risk":
        flagged_file = "flagged_inputs.csv"
        # Save the raw user input, not the one-hot encoded row
        user_row = {
            "Age": age,
            "Saving accounts": saving_account,
            "Checking account": checking_account,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose,
            "Sex": sex,
            "Housing": housing,
            "Job": job
        }
        df_flag = pd.DataFrame([user_row])
        if not os.path.exists(flagged_file):
            df_flag.to_csv(flagged_file, index=False)
        else:
            df_flag.to_csv(flagged_file, mode='a', header=False, index=False)

    # Get SHAP values
    shap_values = get_shap_values(model, input_data)
    # If get_shap_values returns (shap_values, explainer), use: shap_values, explainer = get_shap_values(input_data)

    # Get feature importances for this prediction
    shap_vals = shap_values[0] if hasattr(shap_values, '__getitem__') else shap_values
    feature_importance = sorted(
        zip(input_data.columns, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    # Get top 5 important features
    top_features = feature_importance[:5]
    importance_text = "Top features for this prediction:\n" + "\n".join(
        [f"{feat}: {to_scalar(val):.3f}" for feat, val in top_features]
    )

    return prediction_text, importance_text

# Gradio UI
demo = gr.Interface(
    fn=predict_and_explain,
    inputs=[
        gr.Number(label="Age", info="Customer's Age"),
        gr.Radio(list(saving_account_map.keys()), label="Saving accounts", info="Saving accounts (Amount of Saving)"),
        gr.Radio(list(checking_account_map.keys()), label="Checking account", info="Checking account (in USD)"),
        gr.Number(label="Credit amount", info="Credit amount (numeric, in USD)"),
        gr.Number(label="Duration", info="Duration (numeric, in month)"),
        gr.Radio(purpose_options, label="Purpose", info="Purpose (car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)"),
        gr.Radio(sex_options, label="Sex", info="Sex (Male, Female)"),
        gr.Radio(housing_options, label="Housing", info="Housing (Own, Rent, or Free)"),
        gr.Radio(job_options, label="Job", info="Job (0 - Unskilled and non-resident, 1 - Unskilled and resident, 2 - Skilled, 3 - Highly skilled)")
    ],
    outputs=[
        gr.Text(label="Credit Risk Prediction"),
        gr.Text(label="Top Features for This Prediction")
    ],
    title="Credit Risk Predictor",
    description="Input applicant info to assess credit risk level and see SHAP explanation"
)
 

if __name__ == "__main__":
    demo.launch()
    

