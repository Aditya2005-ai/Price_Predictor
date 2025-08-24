import requests
import pandas as pd

def fetch_products(api_url, params=None):
    """Fetch product data from a public API and return as DataFrame."""
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    data = response.json()
    # If the API returns a list, use it directly
    if isinstance(data, list):
        products = data
    elif isinstance(data, dict) and 'products' in data:
        products = data['products']
    else:
        products = []
    return pd.DataFrame(products)

if __name__ == "__main__":
    # Example API (replace with real one)
    api_url = "https://fakestoreapi.com/products"
    df = fetch_products(api_url)
    df.to_csv("products.csv", index=False)
    print("Scraped and saved products.csv")
