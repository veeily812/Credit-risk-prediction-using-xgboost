import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def get_cleaned_data():
    df = pd.read_csv('German Credit Data.csv')

    # Fill missing values
    df['Saving accounts'] = df['Saving accounts'].fillna('No Savings')
    df['Checking account'] = df['Checking account'].fillna('No Checking')
    df = df.drop(columns='Unnamed: 0')

    #print(df.info())

    num_cols = ['Credit amount', 'Duration in month', 'Age in years']
    cat_cols = ['Saving accounts', 'Checking account', 'Purpose', 'Sex', 'Housing', 'Job']

    #Encoding Categorical Variabpythles
    label = LabelEncoder()
    df['Saving accounts'] = label.fit_transform(df['Saving accounts'])
    df['Checking account'] = label.fit_transform(df['Checking account'])

    #One Hot Encoding
    df = pd.get_dummies(df, columns=['Purpose', 'Sex', 'Housing', 'Job']).astype(int)

    # Scoring system
    risk_score = (
        (df['Credit amount'] > 5000).astype(int) +
        (df['Duration'] > 24).astype(int) +
        (df['Saving accounts'] == 0).astype(int) +  # 0 = 'No Savings' after label encoding
        (df['Checking account'] == 0).astype(int) + # 0 = 'No Checking' after label encoding
        (df['Purpose_radio/TV'] == 1).astype(int) if 'Purpose_radio/TV' in df.columns else 0 +
        (df['Housing_rent'] == 1).astype(int) if 'Housing_rent' in df.columns else 0 +
        (df['Job_0'] == 1).astype(int) if 'Job_0' in df.columns else 0
    )

    # Set threshold: if risk_score >= 3, high risk (1), else low risk (0)
    df['credit_risk'] = (risk_score >= 3).astype(int)
    return df








