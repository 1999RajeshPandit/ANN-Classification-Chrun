import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

st.title('Customer Churn Prediction')
st.write('Enter customer details to predict if they will churn or not.')
credit_score = st.number_input('Credit Score', min_value = 100, max_value = 10000, value = 500)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0.0, value=10000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary],
})
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = input_data.drop('Geography', axis=1)
# input_data_final = np.concatenate((input_data_geo, input_data.drop('Geography', axis=1).values), axis=1)
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data = scaler.transform(input_data)
prediction = model.predict(input_data)
if st.button('Predict'):
    st.write(f'Churn Probability: {prediction[0][0]:.2f}')
    if prediction[0][0] > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')