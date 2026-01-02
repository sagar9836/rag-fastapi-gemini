from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    HF_API_KEY: str

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_ENV: str

    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    class Config:
        env_file = ".env"


settings = Settings()
