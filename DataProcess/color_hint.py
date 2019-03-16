import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


# import tensorflow as tf


def median_pyramid(img, filter_size=2):
    m, n, channels = img.shape[0], img.shape[1], img.shape[2]
    channels = img.shape[2]
    res = np.zeros((m // filter_size, n // filter_size, channels))
    for k in range(channels):
        for i in range(m // filter_size):
            for j in range(n // filter_size):
                res[i, j, k] = np.median(
                    (img[i * filter_size:(i + 1) * filter_size, j * filter_size:(j + 1) * filter_size, k]))
    return res


def generate_color_map(img):
    blurred = median_pyramid(img, 4)
    blurred = cv2.GaussianBlur(blurred, (3, 3), 1)
    # blurred = median_pyramid(blurred, 4)
    # blurred = cv2.GaussianBlur(blurred, (3, 3), 1)
    blurred = median_pyramid(blurred, 4)
    blurred = cv2.GaussianBlur(blurred, (3, 3), 1)
    blurred = cv2.resize(blurred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return blurred.astype(np.uint8)


def generate_whiteout(img, block_shape, block_num):
    if min(block_shape) > min(img.shape[:2]) - 1:
        raise ValueError('Block too large')
    output = img.copy()
    for _ in range(block_num):
        m = random.randint(0, img.shape[0] - block_shape[0])
        n = random.randint(0, img.shape[1] - block_shape[1])
        output[m:m + block_shape[0], n:n + block_shape[1], :] = 255
    return output


def generate_color_block(origin_img, img, block_shape, block_num):
    if not origin_img.shape == img.shape:
        raise ValueError('Original image and input image must have the same shape')
    if min(block_shape) > min(img.shape) - 1:
        raise ValueError('Block too large')
    iter = 0
    output = img.copy()
    while iter < block_num:
        m = random.randint(0, img.shape[0] - block_shape[0])
        n = random.randint(0, img.shape[1] - block_shape[1])
        block = origin_img[m:m + block_shape[0], n:n + block_shape[1], :]
        var = np.sum(np.var(block, axis=(0, 1)))
        # if var:
        #     continue
        output[m:m + block_shape[0], n:n + block_shape[1], :] = np.average(block, axis=(0, 1)).reshape(1, 1, 3)
        iter += 1
    return output


def generate_color_block_random(origin_img, img, block_shape, block_num):
    if not origin_img.shape == img.shape:
        raise ValueError('Original image and input image must have the same shape')
    if min(block_shape) > min(img.shape[:2]) - 1:
        raise ValueError('Shape of original image is ' + str(origin_img.shape) + ', Block too large')

    iter = 0
    output = img.copy()
    while iter < block_num:
        m = random.randint(0, img.shape[0] - block_shape[0])
        n = random.randint(0, img.shape[1] - block_shape[1])
        block = origin_img[m:m + block_shape[0], n:n + block_shape[1], :]
        var = np.sum(np.var(block, axis=(0, 1)))
        # if var:
        #     continue
        output[m:m + block_shape[0], n:n + block_shape[1], :] = np.average(block, axis=(0, 1)).reshape(1, 1, 3)
        iter += 1
    return output


def generate_color_block_random_normal(origin_img, img, block_shape_miu, block_num, block_shape_sigma=6):
    if not origin_img.shape == img.shape:
        raise ValueError('Original image and input image must have the same shape')
    if block_shape_miu + block_shape_sigma > min(img.shape[:2]) - 1:
        raise ValueError('Shape of original image is ' + str(origin_img.shape) + ', Block too large')

    iter = 0
    output = img.copy()
    while iter < block_num:
        block_shape = np.int(np.ceil(abs(np.random.normal(loc=block_shape_miu, scale=block_shape_sigma, size=1)[0])))
        # print(block_shape)
        m = random.randint(0, img.shape[0] - block_shape)
        n = random.randint(0, img.shape[1] - block_shape)
        block = origin_img[m:m + block_shape, n:n + block_shape, :]
        var = np.sum(np.var(block, axis=(0, 1)))
        # if var:
        #     continue
        output[m:m + block_shape, n:n + block_shape, :] = np.average(block, axis=(0, 1)).reshape(1, 1, 3)
        iter += 1
    return output


if __name__ == "__main__":
    img_path = '../demo3.jpg'
    img = cv2.imread(img_path)
    # erosion = cv2.erode(img, np.ones((1, 1)), iterations=2)
    # dilation = cv2.dilate(img, np.ones((3, 3)), iterations=1)
    # edge = dilation - erosion
    # cv2.imshow("", generate_color_map(img))
    # cv2.waitKey()
    img = cv2.resize(img, (512, 512))
    blurred = generate_color_map(img)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    blocked = generate_color_block_random_normal(origin_img=img, img=blurred, block_num=25, block_shape_miu=5,
                                                 block_shape_sigma=6)
    plt.imshow(blocked)
    plt.show()
