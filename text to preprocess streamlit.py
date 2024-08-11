import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
@st.cache_resource
def load_model():
    model_name = "gpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Generate response using GPT-2
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Execute preprocessing step
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

# Streamlit app interface
st.title("Interactive Data Preprocessing Assistant")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())
    
    # User input for preprocessing steps
    st.write("### Describe the preprocessing step you'd like to perform:")
    user_input = st.text_input("You: ")

    if st.button("Generate Preprocessing Suggestion"):
        prompt = f"User wants to preprocess data. {user_input}"
        response = generate_response(prompt, model, tokenizer)
        st.write("### Assistant's Suggestion:")
        st.write(response)

    if st.button("Apply Preprocessing Step"):
        processed_data = execute_preprocessing_step(data, user_input)
        st.write("### Preprocessed Dataset Preview")
        st.write(processed_data.head())
        
        # Option to download the preprocessed data
        st.write("### Download Preprocessed Data")
        csv = processed_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='preprocessed_data.csv', mime='text/csv')

