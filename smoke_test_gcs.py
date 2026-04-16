import os
import json
import requests
from web_app import app
from io import BytesIO

def run_smoke_test():
    print("Starting smoke test for GCS presigned URL generation and upload...")
    
    # Use Flask test client
    app.config['TESTING'] = True
    client = app.test_client()
    
    # 1. Test /api/managed-jobs/upload-url
    print("\n--- Testing /api/managed-jobs/upload-url ---")
    payload = {
        "filename": "test_log_image.tif",
        "contentType": "image/tiff"
    }
    
    response = client.post(
        '/api/managed-jobs/upload-url', 
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to get upload URL. Status: {response.status_code}")
        print("Response:", response.get_data(as_text=True))
        return
        
    data = response.get_json()
    if not data.get("success"):
        print(f"❌ API returned failure: {data.get('error')}")
        return
        
    upload_url = data.get("uploadUrl")
    file_key = data.get("fileKey")
    print(f"✅ Successfully generated presigned URL.")
    print(f"File Key: {file_key}")
    
    # 2. Test actual upload to GCS using the presigned URL
    print("\n--- Testing actual upload to GCS ---")
    test_content = b"This is a fake TIFF image content for smoke testing."
    
    headers = {
        "Content-Type": "image/tiff"
    }
    
    print("Uploading file to GCS...")
    upload_response = requests.put(upload_url, data=test_content, headers=headers)
    
    if upload_response.status_code == 200:
        print("✅ Successfully uploaded file to GCS using the presigned URL!")
    else:
        print(f"❌ Failed to upload to GCS. Status: {upload_response.status_code}")
        print("Response:", upload_response.text)
        return
        
    print("\n🎉 Smoke test passed completely!")

if __name__ == "__main__":
    run_smoke_test()
