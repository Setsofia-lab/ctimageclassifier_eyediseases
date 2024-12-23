import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def prepare_data(data_dir, output_dir, classes, img_height=224, img_width=224, batch_size=32):
    # Create train, val, and test folders
    os.makedirs(output_dir, exist_ok=True)
    for folder in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, folder, cls), exist_ok=True)

    # Split files into train, val, and test sets
    for cls in classes:
        class_path = os.path.join(data_dir, cls)
        files = np.array(os.listdir(class_path))
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        for file_list, folder in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
            for file in file_list:
                shutil.copy(os.path.join(class_path, file), os.path.join(output_dir, folder, cls))

    # Load the datasets
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=(img_height, img_width), batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=(img_height, img_width), batch_size=batch_size
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, image_size=(img_height, img_width), batch_size=batch_size
    )

    return train_ds, val_ds, test_ds
