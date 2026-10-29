"""Скачивание файла токенизатора RWKV World."""
import urllib.request
import os

url = "https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt"
dest = "models/rwkv_vocab_v20230424.txt"
os.makedirs("models", exist_ok=True)
try:
    urllib.request.urlretrieve(url, dest)
    print("  OK")
except Exception as e:
    print(f"  ERROR: {e}")
