from google.cloud import storage
import os
from dotenv import load_dotenv
from pathlib import Path
from google.oauth2 import service_account

load_dotenv()

# Setup credentials exactly as we do in the app
_local_key = Path(__file__).parent / 'GOOGLE_APPLICATION_CREDENTIALS.json'
if _local_key.exists():
    credentials = service_account.Credentials.from_service_account_file(str(_local_key))
    storage_client = storage.Client(credentials=credentials)
else:
    storage_client = storage.Client()

bucket_name = os.getenv('GCS_UPLOADS_BUCKET', 'tiflas-managed-jobs')
bucket = storage_client.get_bucket(bucket_name)

bucket.cors = [
    {
        "origin": ["*"],
        "responseHeader": ["Content-Type", "Access-Control-Allow-Origin", "x-goog-resumable"],
        "method": ["GET", "PUT", "POST", "OPTIONS"],
        "maxAgeSeconds": 3600
    }
]
bucket.patch()
print(f"✅ Successfully set CORS policies for Google Cloud bucket: {bucket.name}")
