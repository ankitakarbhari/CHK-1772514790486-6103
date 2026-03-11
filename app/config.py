import os


class Settings:

    PROJECT_NAME = "Deepfake Detection System"

    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/deepfake_db"
    )

    MODEL_PATH = "models/"


settings = Settings()