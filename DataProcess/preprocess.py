import numpy as np
import cv2
# import matplotlib.pyplot as plt
from keras.models import load_model
import os
from scipy import ndimage
from DataProcess.color_hint import *
from keras import backend as K


def sketch_keras(batch_img):  # shape:(n, 512, 512, 3)
    """
    extract sketches from a batch of images
    :param batch_img: a batch of images
    :return: a batch of sketches
    """
    n = len(batch_img)
    model = load_model('../models/sketchKeras.h5')
    sketches = []
    for i in range(n):
        raw = batch_img[i].astype(np.float32)
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
        sketches.append(sketch)
    K.clear_session()
    return sketches


# def sketch_hed(raw_img):
#     model = load_model('../models/hed.hdf5')
#     raw = cv2.resize(raw_image, (480, 480))
#     raw = np.expand_dims(raw, 0)
#     mask = np.zeros((480, 480))
#     prediction = model.predict(raw)
#     for i in range(len(prediction)):
#         mask += np.reshape(prediction[i], (480, 480))
#     return sketch


def sketch_extract(base_path, img_file_path, out_dir, mod='sketchKeras', resize_shape=(512, 512), batch_size=128):
    """
    extract sketch of images from given directory
    :param base_path: the directory of raw images
    :param img_file_path: the path of file which store the image path (conclude name) in the base_path
    :param out_dir: where to store the sketches
    :param mod: way to extract sketches
    :param resize_shape: the shape of output sketches
    :param batch_size: batch size
    :return: None
    """
    with open(img_file_path, 'r') as f:
        images_path = f.readlines()
    batch_img = []
    count = 0
    total = 0
    names = []
    for i, path in enumerate(images_path):
        img_path = os.path.join(base_path, path.strip('\n'))
        img_name = path.split('/')[-1]
        raw = cv2.imread(img_path)
        raw = cv2.resize(raw, resize_shape)

        batch_img.append(raw)
        names.append(img_name)
        count += 1
        if (i + 1) % batch_size != 0:
            if i + 1 < len(images_path):
                continue
        print('*' * 20, 'Get {:d} images for extraction'.format(count), '*' * 20)
        if mod == 'cv':
            pass
        else:
            if mod == 'hed':
                sketch = sketch_keras(raw)

            elif mod == 'sketchKeras':
                sketch = sketch_keras(batch_img)
            else:
                raise ValueError('Model Name Undefined')
        for j in range(count):
            cv2.imwrite(os.path.join(out_dir, names[j].split('.')[0] + '_sketch.jpg'), sketch[j])
            print('Saved [{}]'.format(os.path.join(out_dir, names[j].split('.')[0] + '_sketch.jpg')))
        total += count
        print('*' * 20, 'Saved {:d} sketches, total saved {:d}'.format(count, total), '*' * 20)
        batch_img = []
        count = 0
        names = []


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

    base_path = '../dataset/raw_image'  # where the raw images locate
    sketch_output = '../dataset/sketch'  # directory to store sketches
    color_hint_output = '../dataset/color_hint'  # directory to store color hint(blur)
    color_hint_whiteout_output = '../dataset/color_hint_with_whiteout'  # directory to store color hint added whiteout
    color_block_output = '../dataset/color_block'  # directory to store color block
    resize_shape = (512, 512)  # raw images should be resized
    for i in range(1, 2, 1):
        list_file = '../dataset/image_list_{:d}.txt'.format(i)  # get image list

        # sketch extraction
        # sketch_output_each = os.path.join(sketch_output, '{:03d}'.format(i))  # store into ".../00?/"
        # if not os.path.exists(sketch_output_each):
        #     os.mkdir(sketch_output_each)
        # sketch_extract(base_path, list_file, sketch_output_each, resize_shape=resize_shape, batch_size=500)
        # print('Extract sketches from ', list_file)

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
            img_names = f.readlines()  # get all images' name
        for i, name in enumerate(img_names):
            img_name = name.strip('\n')
            raw_image = cv2.imread(os.path.join(base_path, img_name))
            raw_image = cv2.resize(raw_image, resize_shape)

            img_num = img_name.split('/')[-1].split('.')[0]  # 'str'

            color_hint = generate_color_map(raw_image)
            cv2.imwrite(os.path.join(color_hint_output_each, img_num + '_colorhint.jpg'), color_hint)
            print('Saved [{}]'.format(os.path.join(color_hint_output_each, img_num + '_colorhint.jpg')),
                  '--no.{}'.format(i + 1))

            color_hint_whiteout = generate_whiteout(color_hint, (10, 10), 10)  # to be determined
            cv2.imwrite(os.path.join(color_hint_whiteout_output_each, img_num + '_whiteout.jpg'), color_hint_whiteout)
            print('Saved [{}]'.format(os.path.join(color_hint_whiteout_output_each, img_num + '_whiteout.jpg')),
                  '--no.{}'.format(i + 1))

            white_canvas = np.ones_like(raw_image) * 255
            color_block = generate_color_block(raw_image, white_canvas, (10, 10), 15)  # to be determined
            cv2.imwrite(os.path.join(color_block_output_each, img_num + '_colorblock.jpg'), color_block)
            print('Saved [{}]'.format(os.path.join(color_block_output_each, img_num + '_colorblock.jpg')),
                  '--no.{}'.format(i + 1))
        print('*' * 10 + 'Saved [{}]'.format(len(img_names)) + ' from ', list_file + '*' * 10)
