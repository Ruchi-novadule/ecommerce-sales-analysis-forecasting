from sklearn.linear_model import LinearRegression

def train_model(df):

    X = df[["Month","Quantity","Price"]]
    y = df["TotalSales"]

    model = LinearRegression()
    model.fit(X,y)

    return model


def predict_future_sales(model):

    import pandas as pd

    future_data = pd.DataFrame({
        "Month":[4,5,6],
        "Quantity":[10,12,15],
        "Price":[200,200,200]
    })

    predictions = model.predict(future_data)

    return predictions