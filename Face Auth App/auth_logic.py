import numpy as np
from face_processing import get_face_embedding_dlib
from database import save_user, get_user, verify_password


def register_user(login: str, password: str, image: np.ndarray) -> bool:
    """
    Реєстрація користувача: логін, пароль, обличчя.
    :param login: логін користувача
    :param password: пароль користувача
    :param image: зображення з обличчям (BGR, np.ndarray)
    :return: True — успішно, False — логін вже зайнятий
    """
    embedding = get_face_embedding_dlib(image)
    if embedding is None:
        print("[!] Обличчя не знайдено або не вдалося створити ембедінг.")
        return False

    return save_user(login, password, embedding)


def login_with_password(login: str, password: str) -> bool:
    """
    Авторизація за паролем.
    :param login: логін
    :param password: введений пароль
    :return: True — успішно
    """
    user = get_user(login)
    if user is None:
        print("[!] Користувача не знайдено.")
        return False

    return verify_password(password, user["password_hash"])


def login_with_face(login: str, image: np.ndarray, threshold=0.6) -> bool:
    """
    Авторизація за обличчям.
    :param login: логін
    :param image: зображення з обличчям
    :param threshold: поріг схожості (чим менше, тим точніше)
    :return: True — успішна авторизація
    """
    user = get_user(login)
    if user is None:
        print("[!] Користувача не знайдено.")
        return False

    embedding_input = get_face_embedding_dlib(image)
    if embedding_input is None:
        print("[!] Не вдалося обробити обличчя.")
        return False

    # Евклідова відстань між ембедінгами
    distance = np.linalg.norm(user["embedding"] - embedding_input)
    print(f"[i] Відстань між обличчями: {distance:.4f}")

    return distance < threshold