import os

def generate_txt_file(image_dir, output_txt):
    with open(output_txt, 'w') as f:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg'):
                    # Write the relative path to the image
                    relative_path = os.path.relpath(os.path.join(root, file), start=image_dir)
                    f.write(f"{relative_path}\n")

if __name__ == "__main__":
    # Paths to the directories containing images
    train_image_dir = 'Dataset/COCOPose/train2017'
    val_image_dir = 'Dataset/COCOPose/val2017'
    
    # Output text files
    train_txt = 'Dataset/COCOPose/train2017.txt'
    val_txt = 'Dataset/COCOPose/val2017.txt'
    
    # Generate the .txt files
    generate_txt_file(train_image_dir, train_txt)
    generate_txt_file(val_image_dir, val_txt)
    
    print("train2017.txt and val2017.txt have been created successfully.")
