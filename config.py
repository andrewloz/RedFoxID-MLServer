import configparser

def config():
    config = configparser.ConfigParser()
    with open("config.ini", "r") as f:
        config.read_file(f)

    models = config["InferenceServer"].get("Models", [])
    models = [m.strip() for m in models.split(",") if m.strip()]

    return config["InferenceServer"], models