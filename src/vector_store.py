import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize vector database
client = chromadb.Client()
collection = client.create_collection(name="sales_data")

def store_dataset_vectors(path):

    df = pd.read_csv(path)

    for i, row in df.iterrows():

        text = f"Product {row['Product']} category {row['Category']} region {row['Region']}"

        embedding = model.encode(text).tolist()

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[text]
        )

    return "Dataset stored in Vector DB"