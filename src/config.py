import configparser

class Config:
    # typical configPath is "config.ini"
    def __init__(self, configPath):
        self.config = configparser.ConfigParser()
        with open(configPath, "r") as f:
            self.config.read_file(f)

        models = self.config["InferenceServer"].get("Models", "")
        self.models = [m.strip() for m in models.split(",") if m.strip()]

        # TODO: add checks here for existance of InferenceServer and Models fields
        # maybe a method that does this checks, hasModels our something.

    def getAll(self):
        return self.config["InferenceServer"], self.models
