import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import os


def sketch_extract(base_path, img_file_path, out_dir, mod='sketchKeras'):
    with open(img_file_path, 'r') as f:
        images_path = f.readlines()
    for i, path in enumerate(images_path):
        img_path = os.path.join(base_path, path)
        raw = cv2.imread(img_path)
        if mod == 'cv':
            pass
        else:
            raw = np.expand_dims(raw, 0)
            if mod == 'hed':
                model = load_model('./models/hed.hdf5')
                prediction = model.predict(raw)

            elif mod == 'sketchKeras':
                model = load_model('./models/sketchKeras.h5')
                img = raw.transpose((3, 1, 2, 0))
                prediction = model.predict(raw)
                sketch_map = prediction.transpose()
            else:
                raise ValueError('Model Name Undefined')


if __name__ == "__main__":
    pass
