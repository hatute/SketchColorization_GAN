import cv2
import numpy as np
import os


def generate_img_file(img_dir, save_file, threshold=10):
    name_list = []
    img_names = os.listdir(img_dir)
    for name in img_names:
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        s_w, s_h = s.shape[:2]
        s_sum = np.sum(s) / (s_w * s_h)
        if s_sum < threshold or img.shape[0] < 512 or img.shape[1] < 512:
            continue
        name_list.append(name)
    with open(save_file, 'w') as f:
        f.write(str(name_list))
        print('Save images\' name to ', save_file)


if __name__ == "__main__":
    generate_img_file('./', './')
