from google import genai

client = genai.Client(api_key="AQ.Ab8RN6J7V6h1UCyiO9nQVLXxPgo44haYCw_7sw4w2gppLvE0UA")

for model in client.models.list():
    print(model.name)