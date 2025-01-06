import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data(file_path=r'C:/Crop Yield Impact/data.csv/pesticides.csv'):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        st.error(f"Error: The file at {file_path} is empty.")
    except pd.errors.ParserError:
        st.error(f"Error: There was a parsing error while reading the file at {file_path}.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Function to train the model
def train_model(df):
    df['Yield'] = df['Value'] * np.random.uniform(0.8, 1.2, df.shape[0])
    X = df[['Value']]
    y = df['Yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model, mse, r2

# Function to visualize the model
def visualize_model(df, model):
    df['Yield'] = df['Value'] * np.random.uniform(0.8, 1.2, df.shape[0])
    X = df[['Value']]
    fig, ax = plt.subplots()
    ax.scatter(X, df['Yield'], color='blue', label='Data Points')
    ax.plot(X, model.predict(X), color='red', label='Regression Line')
    ax.set_xlabel('Pesticides Use (Value)')
    ax.set_ylabel('Yield')
    ax.set_title('Linear Regression')
    ax.legend()
    st.pyplot(fig)

# Function to make predictions
def predict(pesticides_use):
    with open("linear_model.pkl", "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([[pesticides_use]])
    return prediction[0]

# Load and train the model
df = load_data()
if df is not None:
    model, mse, r2 = train_model(df)

    # Streamlit app
    st.title("🌱 Crop Yield Impact Predictor")

    # Create input field for pesticide quantity
    pesticide_quantity = st.number_input("Enter the amount of pesticides used (tonnes)", min_value=0.0, step=0.1, format="%.1f")

    # Display prediction result
    if st.button("Predict Yield"):
        predicted_yield = predict(pesticide_quantity)
        st.success(f"Predicted Yield for {pesticide_quantity:.1f} tonnes of pesticides: {predicted_yield:.2f} units")
        st.balloons()

    # Visualize the model
    st.markdown("### Regression Model Visualization")
    visualize_model(df, model)

    # Display model performance metrics
    st.markdown("### Model Performance")
    st.write(f"**Mean Squared Error**: {mse:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")

    # Add some additional styling
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("Failed to load data. Please check the file path and try again.")
