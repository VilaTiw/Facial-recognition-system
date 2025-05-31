from pymongo import MongoClient
import bcrypt
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users = db[COLLECTION_NAME]

def create_user(login, password, face_embedding):
    if users.find_one({"login": login}):
        return False, "User already exists"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "login": login,
        "password": hashed_pw,
        "embedding": face_embedding.tolist()
    })
    return True, "User registered"

def verify_password(login, password):
    user = users.find_one({"login": login})
    if not user:
        return False
    return bcrypt.checkpw(password.encode(), user["password"])

def get_embedding(login):
    user = users.find_one({"login": login})
    if not user:
        return None
    return user["embedding"]