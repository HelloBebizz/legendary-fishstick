import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_dataset(filepath):
       return pd.read_csv(filepath)

filepath = "X:/Rendumshet/car data.csv"
data = load_dataset(filepath)
#print(data.head())

def load_model():
    model_name = "gpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def interact_with_user(model, tokenizer):
    print("Welcome to the interactive data preprocessing assistant.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        prompt = f"User wants to preprocess data. {user_input}"
        response = generate_response(prompt, model, tokenizer)
        print("Assistant:", response)

def execute_preprocessing_step(df, step_description):
    if "fill missing values with mean" in step_description:
        imputer = SimpleImputer(strategy='mean')
        df[df.columns] = imputer.fit_transform(df)
    elif "fill missing values with median" in step_description:
        imputer = SimpleImputer(strategy='median')
        df[df.columns] = imputer.fit_transform(df)
    elif "fill missing values with mode" in step_description:
        imputer = SimpleImputer(strategy='most_frequent')
        df[df.columns] = imputer.fit_transform(df)
    elif "forward fill" in step_description:
        df = df.fillna(method='ffill')
    elif "backward fill" in step_description:
        df = df.fillna(method='bfill')
    elif "drop rows with missing values" in step_description:
        df = df.dropna()
    elif "drop columns with missing values" in step_description:
        df = df.dropna(axis=1)
    elif "label encoding" in step_description:
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
    elif "one-hot encoding" in step_description:
        df = pd.get_dummies(df, drop_first=True)
    elif "standard scaling" in step_description:
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df)
    elif "min-max scaling" in step_description:
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df)
    elif "robust scaling" in step_description:
        scaler = RobustScaler()
        df[df.columns] = scaler.fit_transform(df)
    elif "log transformation" in step_description:
        df = df.applymap(lambda x: np.log(x) if np.isfinite(x) and x > 0 else x)
    elif "square root transformation" in step_description:
        df = df.applymap(lambda x: np.sqrt(x) if np.isfinite(x) and x >= 0 else x)
    elif "remove outliers" in step_description:
        df = df[(np.abs(df - df.mean()) <= (3*df.std())).all(axis=1)]
    return df

def output_preprocessed_data(df, filepath='preprocessed_data.csv'):
    df.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}")


