import os
from cryptography.fernet import Fernet
from dotenv import set_key, load_dotenv


def encrypt_api_keys(settings_dict: dict):

    load_dotenv()
    print(f"\n\nFERNET_KEY (Encryption): {os.getenv('FERNET_KEY')}\n\n")

    # Check if an auth key is set in the environment variables
    if "FERNET_KEY" in os.environ:
        auth_key = os.environ["FERNET_KEY"]
    else:  # create one
        auth_key = Fernet.generate_key().decode("utf-8")
        set_key(".env", "FERNET_KEY", auth_key)

    # Encode all api keys
    f = Fernet(auth_key.encode("utf-8"))
    if "API_KEYS" in settings_dict:
        for key in settings_dict["API_KEYS"]:
            settings_dict["API_KEYS"][key] = f.encrypt(
                settings_dict["API_KEYS"][key].encode("utf-8")
            ).decode("utf-8")

    if "WCD_API_KEY" in settings_dict:
        settings_dict["WCD_API_KEY"] = f.encrypt(
            settings_dict["WCD_API_KEY"].encode("utf-8")
        ).decode("utf-8")

    return settings_dict


def decrypt_api_keys(settings_dict: dict):

    load_dotenv()
    print(f"\n\nFERNET_KEY (Decryption): {os.getenv('FERNET_KEY')}\n\n")

    if "FERNET_KEY" in os.environ:
        auth_key = os.environ["FERNET_KEY"]
    else:  # create one
        auth_key = Fernet.generate_key().decode("utf-8")
        set_key(".env", "FERNET_KEY", auth_key)

    # decode all api keys
    f = Fernet(auth_key.encode("utf-8"))

    if "API_KEYS" in settings_dict:
        for key in settings_dict["API_KEYS"]:
            settings_dict["API_KEYS"][key] = f.decrypt(
                settings_dict["API_KEYS"][key].encode("utf-8")
            ).decode("utf-8")

    if "WCD_API_KEY" in settings_dict:
        settings_dict["WCD_API_KEY"] = f.decrypt(
            settings_dict["WCD_API_KEY"].encode("utf-8")
        ).decode("utf-8")

    return settings_dict
