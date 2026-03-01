from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Defaults make local + cloud deploy easier
    log_path: str = "logs/api.log"
    diabetes_model_path: str = "model_dir/diabetes_prediction_pipeline.joblib"
    heart_disease_model_path: str = "model_dir/heart_disease_prediction_pipeline.joblib"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()
