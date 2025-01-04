import pickle

def load_model():
    with open("linear_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict(pesticides_use):
    model = load_model()
    prediction = model.predict([[pesticides_use]])
    return prediction[0]

# Example usage
if __name__ == "__main__":
    pesticides_amount = 300  # Example input
    predicted_yield = predict(pesticides_amount)
    print(f"Predicted Yield for {pesticides_amount} units of pesticides: {predicted_yield:.2f} units")
