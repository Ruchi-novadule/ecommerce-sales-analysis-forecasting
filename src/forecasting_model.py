from sklearn.linear_model import LinearRegression
from src.logger import log_exception
import pandas as pd

def train_model(df):
    try:
        X = df[["Month","Quantity","Price"]]
        y = df["TotalSales"]

        model = LinearRegression()
        model.fit(X, y)

        return model

    except Exception as e:
        log_exception(e)
        print("Error occurred while training the model")


def predict_future_sales(model):
    try:
        future_data = pd.DataFrame({
            "Month":[4,5,6],
            "Quantity":[10,12,15],
            "Price":[200,200,200]
        })

        predictions = model.predict(future_data)

        return predictions

    except Exception as e:
        log_exception(e)
        print("Error occurred during prediction")