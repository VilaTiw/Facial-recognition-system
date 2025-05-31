from pymongo import MongoClient
from bson.binary import Binary
import numpy as np
import hashlib
import os

# Підключення до локальної бази MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["face_auth_app"]
users_collection = db["users"]

def hash_password(password: str) -> str:
    """Хешування пароля із використанням salt (SHA-256)."""
    salt = os.urandom(16)
    salted_password = salt + password.encode("utf-8")
    hashed = hashlib.sha256(salted_password).hexdigest()
    return salt.hex() + ":" + hashed

def verify_password(input_password: str, stored_hash: str) -> bool:
    """Перевірка правильності введеного пароля."""
    salt_hex, hash_val = stored_hash.split(":")
    salt = bytes.fromhex(salt_hex)
    input_hashed = hashlib.sha256(salt + input_password.encode("utf-8")).hexdigest()
    return input_hashed == hash_val

def save_user(login: str, password: str, face_embedding: np.ndarray) -> bool:
    """Збереження нового користувача у базі."""
    if users_collection.find_one({"login": login}):
        return False  # Користувач з таким логіном вже існує

    hashed_password = hash_password(password)
    embedding_bytes = Binary(face_embedding.tobytes())

    users_collection.insert_one({
        "login": login,
        "password_hash": hashed_password,
        "embedding": embedding_bytes
    })
    return True

def get_user(login: str):
    """Отримання інформації про користувача за логіном."""
    user = users_collection.find_one({"login": login})
    if user:
        embedding = np.frombuffer(user["embedding"], dtype=np.float64)
        return {
            "login": user["login"],
            "password_hash": user["password_hash"],
            "embedding": embedding
        }
    return None