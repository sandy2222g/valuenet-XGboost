import streamlit as st
import requests
import pandas as pd

st.set_page_config(layout="wide")

st.title("Boston House Price Prediction")

st.write("""
This app predicts the price of a house in Boston based on its features. 
The prediction is made by a backend API that serves a trained XGBoost model.
""")

# Sidebar with input fields
st.sidebar.header("Input Features")

def user_input_features():
    crim = st.sidebar.slider('Criminality Rate (CRIM)', 0.0, 90.0, 0.00632)
    zn = st.sidebar.slider('Proportion of Residential Land (ZN)', 0.0, 100.0, 18.0)
    indus = st.sidebar.slider('Proportion of Non-Retail Business Acres (INDUS)', 0.0, 30.0, 2.31)
    chas = st.sidebar.selectbox('Borders Charles River (CHAS)', [0, 1], 0)
    nox = st.sidebar.slider('Nitric Oxides Concentration (NOX)', 0.0, 1.0, 0.538)
    rm = st.sidebar.slider('Average Number of Rooms (RM)', 3.0, 9.0, 6.575)
    age = st.sidebar.slider('Proportion of Owner-Occupied Units built prior to 1940 (AGE)', 0.0, 100.0, 65.2)
    dis = st.sidebar.slider('Weighted Distances to Boston Employment Centres (DIS)', 1.0, 13.0, 4.09)
    rad = st.sidebar.slider('Index of Accessibility to Radial Highways (RAD)', 1.0, 24.0, 1.0)
    tax = st.sidebar.slider('Property-Tax Rate (TAX)', 180.0, 720.0, 296.0)
    ptratio = st.sidebar.slider('Pupil-Teacher Ratio (PTRATIO)', 12.0, 22.0, 15.3)
    b = st.sidebar.slider('Proportion of People of African American Descent (B)', 0.0, 400.0, 396.9)
    lstat = st.sidebar.slider('Percentage of Lower Status of the Population (LSTAT)', 1.0, 40.0, 4.98)
    
    data = {'crim': crim, 'zn': zn, 'indus': indus, 'chas': chas,
            'nox': nox, 'rm': rm, 'age': age, 'dis': dis,
            'rad': rad, 'tax': tax, 'ptratio': ptratio, 'b': b,
            'lstat': lstat}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input
st.subheader('User Input Features')
st.dataframe(input_df)

# Prediction button
if st.button('Predict Price'):
    # The URL for the backend service as defined in docker-compose.yml
    backend_url = "http://backend:8000/predict/"
    
    try:
        response = requests.post(backend_url, json=input_df.to_dict(orient='records')[0])
        response.raise_for_status()  # Raise an exception for bad status codes
        
        prediction = response.json()
        price = prediction.get('predicted_price', 'N/A')
        
        st.success(f"The predicted house price is ${price:,.2f} (in thousands)")
        
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend service. Please ensure the backend container is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the request: {e}")
        st.error(f"Response content: {response.text}")

