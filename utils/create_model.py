from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from utils.data import get_cleaned_data
   
def create_model():
    df = get_cleaned_data()
    #feature selection
    X = df.drop('credit_risk', axis=1)
    y = df['credit_risk']


    #split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    #train model
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)

    #evaluate model
    y_pred = model.predict(X_test)

    #print("classification report /n:", classification_report(y_test, y_pred)) 
    #print("R2", r2_score(y_test, y_pred))
   # print("MSE:", mean_squared_error(y_test, y_pred))
    return model, scaler