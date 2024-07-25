from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    # Load and prepare the data
    data = pd.read_csv(r"C:\Users\afrin\Downloads\diabetes (1).csv")
    x = data.drop("Outcome", axis=1)
    y = data['Outcome']

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Get input data from request and convert to float
    try:
        vals = [
            float(request.GET.get('n1', 0)),
            float(request.GET.get('n2', 0)),
            float(request.GET.get('n3', 0)),
            float(request.GET.get('n4', 0)),
            float(request.GET.get('n5', 0)),
            float(request.GET.get('n6', 0)),
            float(request.GET.get('n7', 0)),
            float(request.GET.get('n8', 0))
        ]
    except ValueError:
        return render(request, "predict.html", {"result2": "Invalid input. Please provide valid numbers."})

    # Convert input data to 2D array
    input_data = np.array(vals).reshape(1, -1)

    # Make the prediction
    pred = model.predict(input_data)

    # Map prediction to result
    result1 = "Positive" if pred[0] == 1 else "Negative"

    return render(request, "predict.html", {"result2": result1})
