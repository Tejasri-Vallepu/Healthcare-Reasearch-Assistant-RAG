from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()


models = client.models.list()


print("Available models:\n")

for model in models.data:
    print(model.id)