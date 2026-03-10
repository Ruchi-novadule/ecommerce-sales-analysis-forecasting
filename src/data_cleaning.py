import pandas as pd

def load_and_clean_data(path):
    
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