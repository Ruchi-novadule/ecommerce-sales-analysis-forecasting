import pandas as pd
from src.logger import log_exception

def load_and_clean_data(path):
    try:
        df = pd.read_csv(path)

        # Convert date column
        df["OrderDate"] = pd.to_datetime(df["OrderDate"])

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Create new columns
        df["TotalSales"] = df["Quantity"] * df["Price"]
        df["Year"] = df["OrderDate"].dt.year
        df["Month"] = df["OrderDate"].dt.month

        return df

    except Exception as e:
        log_exception(e)
        print("Error occurred while loading or cleaning data")