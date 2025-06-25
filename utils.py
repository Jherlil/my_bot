import yaml
from datetime import datetime

def log(message: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}")

def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
