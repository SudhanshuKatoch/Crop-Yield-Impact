# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import logging

# Configure logging
logging.basicConfig(filename='model_performance.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the dataset
def load_data(file_path='C:/Organic Farm Yield/data.csv/pesticides.csv'):
    df = pd.read_csv(file_path)
    print("Columns in the dataset:", df.columns)  # Print column names to verify
    return df

def train_model():
    # Load data
    df = load_data()

    # Print first few rows to verify data
    print(df.head())

    # Creating an artificial 'Yield' column for demonstration
    df['Yield'] = df['Value'] * np.random.uniform(0.8, 1.2, df.shape[0])

    # Features and target
    X = df[['Value']]
    y = df['Yield']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:\nMean Squared Error: {mse:.2f}\nR² Score: {r2:.2f}")

    # Log the performance metrics
    logging.info(f'Mean Squared Error: {mse:.2f}')
    logging.info(f'R² Score: {r2:.2f}')

    # Save the model
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Linear Regression model saved successfully!")

    # Visualize the regression line
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='red', label='Regression Line')
    plt.xlabel('Pesticides Use (Value)')
    plt.ylabel('Yield')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()

