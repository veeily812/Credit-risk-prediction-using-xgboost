import shap
import pickle
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def get_shap_values(model, input_data):
    """
    Returns SHAP values for the given input DataFrame using the trained model.
    input_df: pandas DataFrame, preprocessed and matching model input columns
    Returns: shap_values, explainer
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    return shap_values, explainer



