import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='NeoJaundice', type=str, help='name of dataset')
args = parser.parse_args()
dataset = args.dataset


dataset_folder = f'../medical/{dataset}/origin'
train_folder = f'../medical/{dataset}/train'
test_folder = f'../medical/{dataset}/test'

subfolders = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for subfolder in subfolders:
    label = os.path.basename(subfolder)
    train_label_folder = os.path.join(train_folder, label)
    test_label_folder = os.path.join(test_folder, label)

    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    images = [f.path for f in os.scandir(os.path.join(subfolder, 'images')) if f.is_file()]

    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    for img_path in train_images:
        shutil.copy(img_path, os.path.join(train_label_folder, os.path.basename(img_path)))

    for img_path in test_images:
        shutil.copy(img_path, os.path.join(test_label_folder, os.path.basename(img_path)))