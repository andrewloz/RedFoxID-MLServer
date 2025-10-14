import configparser

def config():
    config = configparser.ConfigParser()
    with open("config.ini", "r") as f:
        config.read_file(f)
    return config