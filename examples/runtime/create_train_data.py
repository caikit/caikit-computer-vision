"""Creates a directory of garbage data to be used for training; this is only
used to demonstrate the training interface for caikit runtime. Since the
.train method of the transformer based object detector is currently just a
stub, the training data contents are not used, and are only for demo purposes.
"""
from common import TRAINING_DATA_DIR, TRAINING_IMG_DIR, TRAINING_LABELS_FILE
import os
import numpy as np
from PIL import Image
from shutil import rmtree

def init_train_data():
    if os.path.isdir(TRAINING_DATA_DIR):
        rmtree(TRAINING_DATA_DIR)
    os.mkdir(TRAINING_DATA_DIR)
    os.mkdir(TRAINING_IMG_DIR)
    # Just make 10 images
    with open(TRAINING_LABELS_FILE, "w") as train_file:
        for img_num in range(1, 11):
            img_name = f"{img_num}.jpg"
            img, label_info = get_random_image_info(img_name, train_file)
            train_file.write(f"{label_info}\n")
            img.save(os.path.join(TRAINING_IMG_DIR, img_name))

def get_random_image_info(img_name, train_file, delimiter="\t", width=50, height=50):
    arr = np.random.randint(low=0, high=256, dtype=np.uint8, size=(width, height, 3))
    img = Image.fromarray(arr)
    label = np.random.choice(["cat", "dog"])
    coords = get_random_box(width, height)
    label_components = [img_name, label, coords]
    label_info = delimiter.join(label_components)
    return img, label_info

def get_random_box(width, height):
    x_min = np.random.randint(0, width-1)
    x_max = np.random.randint(x_min+1, width)
    y_min = np.random.randint(0, height-1)
    y_max = np.random.randint(y_min+1, height)
    return f"({x_min}, {x_max}, {y_min}, {y_max})"

if __name__ == "__main__":
    init_train_data()
