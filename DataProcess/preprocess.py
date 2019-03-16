import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from scipy import ndimage
from DataProcess.color_hint import *


def sketch_extract(base_path, img_file_path, out_dir, mod='sketchKeras'):
    with open(img_file_path, 'r') as f:
        images_path = f.readlines()
    for i, path in enumerate(images_path):
        img_path = os.path.join(base_path, path)
        img_name = path.split('/')[-1]
        raw = cv2.imread(img_path)
        if mod == 'cv':
            pass
        else:
            if mod == 'hed':
                model = load_model('./models/hed.hdf5')
                raw = np.expand_dims(raw, 0)
                prediction = model.predict(raw)

            elif mod == 'sketchKeras':
                model = load_model('./models/sketchKeras.h5')
                raw = raw.astype(np.float32)
                light_map = np.zeros_like(raw).astype(np.float32)
                for channel in range(light_map.shape[2]):  # get high pass map
                    light_map[:, :, channel] = (raw[:, :, channel] - cv2.GaussianBlur(raw[:, :, channel], (0, 0),
                                                                                      3)) / 128
                light_map = light_map / np.max(light_map)  # normalize
                input_mat = np.expand_dims(light_map, 0).transpose((3, 1, 2, 0))  # input shape: (3, 512, 512, 1)
                prediction = model.predict(input_mat)  # output shape: (3, 512, 512, 1)
                sketch_map = prediction.transpose((3, 1, 2, 0))[0]  # (512, 152, 3)
                sketch_map = np.amax(sketch_map, axis=2)  # (512, 512)
                # get output sketch
                sketch_map[sketch_map < 0.18] = 0
                sketch_map = (1 - sketch_map) * 255
                sketch_map = np.clip(sketch_map, a_min=0, a_max=255).astype(np.uint8)
                sketch = ndimage.median_filter(sketch_map, 1)
            else:
                raise ValueError('Model Name Undefined')
        cv2.imwrite(os.path.join(out_dir, img_name.split('.')[0] + '_sketch.jpg'), sketch)


if __name__ == "__main__":
    # raw = cv2.imread('../demo3.jpg')
    # raw = cv2.resize(raw, (512, 512))
    # model = load_model('../models/sketckKeras.h5')
    # raw = raw.astype(np.float32)
    # light_map = np.zeros_like(raw).astype(np.float32)
    # for channel in range(light_map.shape[2]):  # get high pass map
    #     light_map[:, :, channel] = (raw[:, :, channel] - cv2.GaussianBlur(raw[:, :, channel], (0, 0),
    #                                                                       3)) / 128
    # light_map = light_map / np.max(light_map)  # normalize
    # input_mat = np.expand_dims(light_map, 0).transpose((3, 1, 2, 0))  # input shape: (3, 512, 512, 1)
    # prediction = model.predict(input_mat)  # output shape: (3, 512, 512, 1)
    # sketch_map = prediction.transpose((3, 1, 2, 0))[0]  # (512, 152, 3)
    # sketch_map = np.amax(sketch_map, axis=2)  # (512, 512)
    # # get output sketch
    # sketch_map[sketch_map < 0.18] = 0
    # sketch_map = (1 - sketch_map) * 255
    # sketch_map = np.clip(sketch_map, a_min=0, a_max=255).astype(np.uint8)
    # sketch = ndimage.median_filter(sketch_map, 1)
    # plt.imshow(cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB))
    # plt.show()
    # out_dir = './'
    # img_name = '../demo3.jpg'.split('/')[-1]
    # cv2.imwrite(os.path.join(out_dir, img_name.split('.')[0] + '_sketch.jpg'), sketch)

    base_path = ''
    sketch_output = '../dataset/sketch'
    color_hint_output = '../dataset/color_hint'
    color_hint_whiteout_output = '../dataset/color_hint_with_whiteout'
    color_block_output = '../dataset/color_block'
    resize_shape = (256, 256)
    for i in range(1, 9, 1):
        list_file = '../dataset/image_list_{:d}.txt'.format(i)

        # sketch extraction
        sketch_output_each = os.path.join(sketch_output, '{:03d}'.format(i))
        if not os.path.exists(sketch_output_each):
            os.mkdir(sketch_output_each)
        sketch_extract(base_path, list_file, sketch_output_each)

        # color hint, whiteout, color block
        color_hint_output_each = os.path.join(color_hint_output, '{:03d}'.format(i))
        if not os.path.exists(color_hint_output_each):
            os.mkdir(color_hint_output_each)

        color_hint_whiteout_output_each = os.path.join(color_hint_whiteout_output, '{:03d}'.format(i))
        if not os.path.exists(color_hint_whiteout_output_each):
            os.mkdir(color_hint_whiteout_output_each)

        color_block_output_each = os.path.join(color_block_output, '{:03d}'.format(i))
        if not os.path.exists(color_block_output_each):
            os.mkdir(color_block_output_each)

        with open(list_file) as f:
            img_names = f.readlines()
        for name in img_names:
            img_name = name.strip('\n')
            raw_image = cv2.imread(os.path.join(base_path, img_name))
            raw_image = cv2.resize(raw_image, resize_shape)

            img_num = img_name.split('/')[-1].split('.')[0]  # 'str'
            color_hint = generate_color_map(raw_image)
            cv2.imwrite(os.path.join(color_hint_output_each, img_num + '_colorhint.jpg'), color_hint)

            color_hint_whiteout = generate_whiteout(color_hint, (), 0)
            cv2.imwrite(os.path.join(color_hint_whiteout_output_each, img_num + '_whiteout.jpg'), color_hint_whiteout)

            white_canvas = np.ones_like(raw_image) * 255
            color_block = generate_color_block(raw_image, white_canvas, (), 0)
            cv2.imwrite(os.path.join(color_block_output_each, img_num + '_colorblock.jpg'), color_block)
