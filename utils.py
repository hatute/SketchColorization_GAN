import cv2
import numpy as np
import os
import time


def generate_img_file(img_dir, save_path, threshold=10):
    name_list = []
    img_names = os.listdir(img_dir)
    count = 0
    pre = time.time()
    for name in img_names:
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue
        if min(img.shape[0], img.shape[1]) < 512:
            continue
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        s_w, s_h = s.shape[:2]
        s_sum = np.sum(s) / (s_w * s_h)
        if s_sum < threshold or img.shape[0] < 512 or img.shape[1] < 512:
            continue
        name_list.append(name)
        count += 1
        print('Image [{}] saved, already saved {:d}'.format(name, count))
        if count % 10000 == 0 and count > 0:
            with open(os.path.join(save_path, '{:d}.txt'.format(int(count // 10000))), 'w') as f:
                f.write(str(name_list))
                print('*' * 20, 'Save images\' names to {:d}.txt'.format(int(count // 10000)),
                      ' ({:d}--{:d})'.format(count - 10000, count), '*' * 20)
                used = time.time() - pre
                print('Using {:d}min {:d}s'.format(used // 60, used % 60))
                pre = time.time()
            name_list = []
    if count % 10000 != 0:
        with open(os.path.join(save_path, '{:d}.txt'.format(int(count // 10000))), 'w') as f:
            f.write(str(name_list))
            print('*' * 20, 'Save images\' names to {:d}.txt'.format(int(count // 10000)),
                  ' ({:d}--{:d})'.format(count - 10000, count), '*' * 20)


if __name__ == "__main__":
    generate_img_file('./', './dataset')
