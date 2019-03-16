import cv2
import numpy as np
import os
import time
import shutil


def generate_img_file(img_dir, save_path, threshold=10):
    name_list = []
    img_names = os.listdir(img_dir)
    count = 0
    pre = time.time()
    broken_imgs = []
    for name in img_names:
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            print('Passed [{}]--Broken'.format(name))
            broken_imgs.append(name)
            continue
        if min(img.shape[0], img.shape[1]) < 256:
            print('Passed [{}]--Small'.format(name))
            continue
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        s_w, s_h = s.shape[:2]
        s_sum = np.sum(s) / (s_w * s_h)
        if s_sum < threshold:
            print('Passed [{}]--Is a Sketch'.format(name))
            continue
        name_list.append(name)
        count += 1
        print('Image [{}] saved, already saved {:d}'.format(name, count))
        if count % 10000 == 0 and count > 0:
            with open(os.path.join(save_path, 'image_list_{:d}.txt'.format(int(count // 10000))), 'w') as f:
                for i in name_list:
                    f.write(i)
                    f.write('\n')
                # f.write(str(name_list))
                print('*' * 20, 'Save images\' names to {:d}.txt'.format(int(count // 10000)),
                      ' ({:d}--{:d})'.format(count - 10000, count), '*' * 20)
                used = time.time() - pre
                print('Using {:d}min {:d}s'.format(int(used / 60), int(used % 60)))
                pre = time.time()
            name_list = []
    if count % 10000 != 0:
        with open(os.path.join(save_path, 'image_list_{:d}.txt'.format(int(count // 10000))), 'w') as f:
            for i in name_list:
                f.write(i)
                f.write('\n')
            # f.write(str(name_list))
            print('*' * 20, 'Save images\' names to {:d}.txt'.format(int(count // 10000)),
                  ' ({:d}--{:d})'.format(count - count % 10000, count), '*' * 20)
    for broken in broken_imgs:
        try:
            os.remove(os.path.join(img_dir, broken))
            print('Delete {}'.format(broken))
        except NotImplementedError as e:
            print('Delete {} fail'.format(broken))


def move_files_by_namelist(img_dir, save_path, list_file):
    num = int(list_file.split('_')[2].split('.')[0])
    if not os.path.exists(os.path.join(save_path, '{:03d}'.format(num))):
        os.mkdir(os.path.join(save_path, '{:03d}'.format(num)))
    with open(list_file, 'r+') as f:
        names = f.readlines()
        f.seek(0, 0)
        for name in names:
            new_name = '{:03d}/'.format(num) + name
            shutil.move(os.path.join(img_dir, name.strip('\n')), os.path.join(save_path, '{:03d}'.format(num)))
            f.write(new_name)
            print('Move ', name, ' to ', new_name)


if __name__ == "__main__":
    # generate_img_file('/media/bilin/MyPassport/zerochain', './dataset')
    for i in range(1, 9, 1):
        list_file = './dataset/image_list_{:d}.txt'.format(i)
        move_files_by_namelist('/media/bilin/MyPassport/zerochain', '/media/bilin/MyPassport/zerochain', list_file)
        print('*' * 20, 'Success move images listed in ', list_file, '*' * 20)
