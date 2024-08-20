import os
import requests
import zipfile
import tqdm

def download_file_from_url(url, save_path, chunk_size=1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as file, tqdm.tqdm(
        desc=save_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            bar.update(len(chunk))
            file.write(chunk)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_coco_dataset(destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    # URLs for the COCO 2017 dataset
    urls = {
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    for key, url in urls.items():
        filename = os.path.join(destination_folder, os.path.basename(url))
        
        # Download the file
        print(f"Downloading {key}...")
        download_file_from_url(url, filename)
        
        # Unzip the file
        print(f"Extracting {key}...")
        unzip_file(filename, destination_folder)
        
        # Remove the zip file to save space
        os.remove(filename)
        print(f"{key} downloaded and extracted successfully.\n")

if __name__ == "__main__":
    # Set the destination folder where the dataset will be saved
    dataset_folder = "../Dataset/COCOPose"
    
    # Download and extract the dataset
    download_coco_dataset(dataset_folder)
