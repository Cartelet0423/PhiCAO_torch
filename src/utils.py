import numpy as np
import cv2

def load_dv_as_numpy(file_path, width=256, height=256, header_size=512):
    raw_data = np.fromfile(file_path, dtype='>u2')
    return raw_data[header_size:].reshape((-1, height, width))

def resize_stack(image_stack, size=(128, 128)):
    return np.array([cv2.resize(img, size, interpolation=cv2.INTER_AREA) for img in image_stack])