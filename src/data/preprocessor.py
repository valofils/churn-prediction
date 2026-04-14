"""Feature engineering and preprocessing pipeline."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    return df

def encode_target(df: pd.DataFrame):
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y

def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
