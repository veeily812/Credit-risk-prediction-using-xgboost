# main.py
from utils.data import get_cleaned_data
from utils.create_model import create_model
import pickle

def main():
    df = get_cleaned_data()  # Cleans and returns DataFrame
    model, scaler = create_model()  # Builds and evaluates model

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as g:
        pickle.dump(scaler, g)



if __name__ == "__main__":
    main()
