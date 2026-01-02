import boto3
from app.config import settings


def upload_file_to_s3(file_path: str, filename: str) -> str:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )

    s3.upload_file(
        file_path,
        settings.S3_BUCKET_NAME,
        filename
    )

    return f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{filename}"
