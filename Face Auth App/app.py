import cv2
import bcrypt
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from face_processing import detect_face, get_face_embedding, facetracker

# Налаштування MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["face_auth"]
users = db["users"]


def capture_and_process_face():
    """Захоплення обличчя з вебкамери, повертає ембедінг або None."""
    cap = cv2.VideoCapture(0)
    face_embedding = None

    print("Натисніть 's', щоб зробити фото, або 'q' для виходу.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = frame[135:585, 415:865]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        y_hat = facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = y_hat[1][0]
        scaled_coords = np.multiply(sample_coords, [450, 450, 450, 450]).astype(int)

        if y_hat[0] > 0.5:
            # Main rectangle
            cv2.rectangle(frame,
                          tuple(scaled_coords[:2]),
                          tuple(scaled_coords[2:]),
                          (255, 0, 0), 2)
            # Label rectangle
            cv2.rectangle(frame,
                          tuple(np.add(scaled_coords[:2], [0, -30])),
                          tuple(np.add(scaled_coords[:2], [80, 0])),
                          (255, 0, 0), -1)
            # Label
            cv2.putText(frame, "Face", tuple(np.add(scaled_coords[:2], [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Захоплення обличчя", frame)
        key = cv2.waitKey(1)

        if key == ord("s"):
            face_img, _ = detect_face(frame)
            if face_img is not None:
                embedding = get_face_embedding(face_img)
                if embedding is not None:
                    face_embedding = embedding
                    print("Обличчя оброблено.")
                else:
                    print("Не вдалося отримати вектор обличчя.")
            else:
                print("Обличчя не знайдено.")
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_embedding


def register():
    login = input("Введіть логін: ").strip()
    if users.find_one({"login": login}):
        print("Користувач з таким логіном вже існує.")
        return

    password = input("Введіть пароль: ").strip()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    print("Будь ласка, покажіть обличчя...")
    embedding = capture_and_process_face()

    if embedding is None:
        print("Реєстрація скасована: не вдалося розпізнати обличчя.")
        return

    users.insert_one({
        "login": login,
        "password": hashed_pw,
        "embedding": embedding.tolist()
    })

    print("Реєстрація успішна!")


def login_with_password():
    login = input("Логін: ").strip()
    password = input("Пароль: ").strip()

    user = users.find_one({"login": login})
    if not user or not bcrypt.checkpw(password.encode(), user["password"]):
        print("Невірний логін або пароль.")
        return

    print("Авторизація успішна!")


def login_with_face():
    login = input("Введіть логін: ").strip()
    user = users.find_one({"login": login})

    if not user:
        print("Користувача з таким логіном не знайдено.")
        return

    print("Покажіть обличчя для авторизації...")

    input_embedding = capture_and_process_face()

    if input_embedding is None:
        print("Не вдалося зчитати обличчя.")
        return

    db_embedding = np.array(user["embedding"])
    distance = np.linalg.norm(input_embedding - db_embedding)

    if distance < 0.6:
        print(f"Авторизація успішна! Вітаємо, {login}")
    else:
        print("Обличчя не збігається з зареєстрованим.")


def main():
    while True:
        print("\n=== МЕНЮ ===")
        print("1. Реєстрація")
        print("2. Авторизація (логін + пароль)")
        print("3. Авторизація (обличчя)")
        print("0. Вихід")

        choice = input("Ваш вибір: ").strip()
        if choice == "1":
            register()
        elif choice == "2":
            login_with_password()
        elif choice == "3":
            login_with_face()
        elif choice == "0":
            print("Завершення програми.")
            break
        else:
            print("Невірний вибір. Спробуйте ще.")


if __name__ == "__main__":
    main()