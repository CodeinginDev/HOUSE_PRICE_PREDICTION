import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle as pk

# Load the trained model
model_path = r'C:\Users\Aanand Jha\Desktop\house-price-prediction\house_price_model.pkl'
with open(model_path, 'rb') as file:
    model = pk.load(file)

st.header('Nepal House Price Predictor')

# User Inputs
st.subheader('Enter the details of the house:')

area = st.number_input('Area (in sq. ft.)', min_value=500, max_value=10000, value=2000)
bedrooms = st.slider('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.slider('Number of Bathrooms', min_value=1, max_value=10, value=2)
stories = st.slider('Number of Stories', min_value=1, max_value=5, value=2)
parking = st.slider('Number of Parking Spaces', min_value=0, max_value=5, value=1)

mainroad = st.selectbox('Is it on the main road?', ('Yes', 'No'))
guestroom = st.selectbox('Is there a guest room?', ('Yes', 'No'))
basement = st.selectbox('Does it have a basement?', ('Yes', 'No'))
hotwaterheating = st.selectbox('Does it have hot water heating?', ('Yes', 'No'))
airconditioning = st.selectbox('Does it have air conditioning?', ('Yes', 'No'))
furnishingstatus = st.selectbox('Furnishing Status', ('Furnished', 'Semi-Furnished', 'Unfurnished'))
prefarea = st.selectbox('Is it in a preferred area?', ('Yes', 'No'))

# Encode categorical variables
mainroad = 1 if mainroad == 'Yes' else 0
guestroom = 1 if guestroom == 'Yes' else 0
basement = 1 if basement == 'Yes' else 0
hotwaterheating = 1 if hotwaterheating == 'Yes' else 0
airconditioning = 1 if airconditioning == 'Yes' else 0
prefarea = 1 if prefarea == 'Yes' else 0

furnishingstatus_furnished = 1 if furnishingstatus == 'Furnished' else 0
furnishingstatus_semi_furnished = 1 if furnishingstatus == 'Semi-Furnished' else 0
furnishingstatus_unfurnished = 1 if furnishingstatus == 'Unfurnished' else 0

# Feature vector
features = np.array([area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement,
                     hotwaterheating, airconditioning, prefarea, furnishingstatus_furnished, 
                     furnishingstatus_semi_furnished, furnishingstatus_unfurnished])

# Ensure the correct number of features are provided
if features.shape[0] != 13:
    st.error(f"Expected 13 features, but got {features.shape[0]}. Please check your input.")
else:
    # Normalize the features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features.reshape(1, -1))

    # Predict price
    predicted_price = model.predict(features)[0]

    if st.button('Predict Price'):
        st.subheader(f'Estimated House Price: NPR {predicted_price * 1000000:.2f}')
