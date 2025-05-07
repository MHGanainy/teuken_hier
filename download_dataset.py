import os
import gdown
import zipfile

# Google Drive file ID and destination path
file_id = "1AmOr6caROI0MHOhz2A4cTq5Gi-myyDnu"
output_zip = "data/processed/downloaded.zip"
target_dir = "data/processed"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Construct the download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
print("Downloading from Google Drive...")
gdown.download(url, output_zip, quiet=False)

# Extract the ZIP file
print("Extracting contents...")
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

# Optionally, remove the zip file after extraction
os.remove(output_zip)

print(f"Done! Extracted to {target_dir}")
