import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Sample data generation
customers = [f"C{str(i).zfill(3)}" for i in range(1, 21)]
products = [
    ("Apple Watch", "Electronics", 199),
    ("Nike Shoes", "Fashion", 120),
    ("Laptop", "Electronics", 899),
    ("iPhone Case", "Accessories", 25),
    ("Sunglasses", "Fashion", 75),
    ("T-shirt", "Fashion", 30),
    ("Monitor", "Electronics", 250),
    ("Mouse", "Electronics", 40),
    ("Backpack", "Accessories", 60),
    ("Headphones", "Electronics", 150)
]

def generate_order_data(n=100):
    data = []
    start_date = datetime(2024, 1, 1)
    for i in range(n):
        customer = random.choice(customers)
        product, category, price = random.choice(products)
        quantity = random.randint(1, 4)
        order_date = start_date + timedelta(days=random.randint(0, 120))
        order_id = f"O{str(i+1).zfill(4)}"
        data.append([customer, order_id, order_date.strftime('%Y-%m-%d'), product, category, price, quantity])
    return pd.DataFrame(data, columns=["CustomerID", "OrderID", "OrderDate", "Product", "Category", "Price", "Quantity"])

df = generate_order_data()
df.to_csv("sales_data.csv", index=False)
print("✅ sales_data.csv generated.")
