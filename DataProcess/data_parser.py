import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class DataParser:
    def __init__(self, inputs, batch_size):
        self.inputs = inputs
        self.iterator = 0
        self.batch_size = batch_size
        self.m = inputs.shape[0]
        self.iteration = int(np.ceil(self.m / self.batch_size))
        self.indices = np.random.permutation(self.m)

    def get_batch(self):
        if self.iterator + 1 < self.iteration:
            batch_indices = self.indices[self.iterator * self.batch_size:(self.iterator + 1) * self.batch_size]
            self.iterator += 1
        else:
            batch_indices = self.indices[self.iterator * self.batch_size:]
            self.iterator = 0
            self.indices = np.random.permutation(self.m)
        return self.inputs[batch_indices]


class DataParserV2:
    def __init__(self, dataset_path, resize_shape, list_files, batch_size):
        self.dataset_path = dataset_path
        self.resize_shape = resize_shape
        self.list_files = list_files
        self.batch_size = batch_size

        self.raw_image_path = os.path.join(self.dataset_path, 'raw_image')
        self.sketch_path = os.path.join(self.dataset_path, 'sketch')
        self.color_hint_path = os.path.join(self.dataset_path, 'color_hint')
        self.color_hint_whiteout_path = os.path.join(self.dataset_path, 'color_hint_with_whiteout')
        self.color_block_path = os.path.join(self.dataset_path, '../dataset/color_block')
        self.images_name = []
        for lf in self.list_files:
            with open(lf, 'r') as f:
                name = f.readline()
                while name:
                    self.images_name.append(name.strip('\n'))
                    name = f.readline()
        self.m = len(self.images_name)
        self.iteration = int(np.ceil(self.m / self.batch_size))
        self.iterator = 0
        self.indices = np.random.permutation(self.m)

    def get_indices(self, update=False):
        if self.iterator + 1 < self.iteration:
            batch_indices = self.indices[self.iterator * self.batch_size:(self.iterator + 1) * self.batch_size]
            if update:
                self.iterator += 1
        else:
            batch_indices = self.indices[self.iterator * self.batch_size:]
            if update:
                self.iterator = 0
                self.indices = np.random.permutation(self.m)
        return batch_indices

    def update_iterator(self):
        self.get_indices(True)

    def get_batch_raw(self):
        indices = self.get_indices()
        raws = []
        for id in indices:
            raw_name = os.path.join(self.raw_image_path, self.images_name[id])
            raw_img = cv2.imread(raw_name).astype(np.float32)
            raw_img = cv2.resize(raw_img, self.resize_shape)
            raws.append(1 - raw_img / 255)
        return np.asarray(raws)

    def get_batch_sketch(self):
        indices = self.get_indices()
        sketches = []
        for id in indices:
            sketch_name = os.path.join(self.sketch_path, self.images_name[id].split('.')[0] + '_sketch.jpg')
            sketch = cv2.imread(sketch_name, 0).astype(np.float32)
            sketch = cv2.resize(sketch, self.resize_shape)
            noise = np.random.normal(loc=0, scale=1, size=1000)
            pos = np.random.permutation(self.resize_shape[0] * self.resize_shape[1])[:1000]
            noise_channel = np.zeros_like(sketch)
            for i, val in enumerate(pos):
                noise_channel[val // self.resize_shape[0]][val % self.resize_shape[1]] = noise[i]
            sketch = np.expand_dims(sketch, axis=2)
            noise_channel = np.expand_dims(noise_channel, axis=2)
            sketch = sketch / 255 + noise_channel
            sketches.append(sketch)
        return np.asarray(sketches)

    def get_batch_color_hint(self):
        indices = self.get_indices()
        res = []
        for id in indices:
            name = os.path.join(self.color_hint_path, self.images_name[id].split('.')[0] + '_colorhint.jpg')
            img = cv2.imread(name).astype(np.float32)
            img = cv2.resize(img, self.resize_shape)
            res.append(img / 255)
        return np.asarray(res)

    def get_batch_color_hint_whiteout(self):
        indices = self.get_indices()
        res = []
        for id in indices:
            name = os.path.join(self.color_hint_whiteout_path, self.images_name[id].split('.')[0] + '_whiteout.jpg')
            img = cv2.imread(name).astype(np.float32)
            img = cv2.resize(img, self.resize_shape)
            res.append(1 - img / 255)
        return np.asarray(res)

    def get_batch_color_block(self):
        indices = self.get_indices()
        res = []
        for id in indices:
            name = os.path.join(self.color_block_path, self.images_name[id].split('.')[0] + '_colorblock.jpg')
            img = cv2.imread(name).astype(np.float32)
            img = cv2.resize(img, self.resize_shape)
            res.append(1 - img / 255)
        return np.asarray(res)

    def get_batch_condition(self):
        white_outs = self.get_batch_color_hint_whiteout()
        color_blocks = self.get_batch_color_block()
        return np.concatenate((white_outs, color_blocks), axis=3)

    def get_batch_condition_add(self):
        indices = self.get_indices()
        res = []
        for id in indices:
            name_block = os.path.join(self.color_block_path, self.images_name[id].split('.')[0] + '_colorblock.jpg')
            name_whiteout = os.path.join(self.color_hint_whiteout_path,
                                         self.images_name[id].split('.')[0] + '_whiteout.jpg')
            img = cv2.imread(name_block).astype(np.float32)
            img_block = cv2.resize(img, self.resize_shape)
            img_block = 1 - img_block / 255

            img = cv2.imread(name_whiteout).astype(np.float32)
            img_whiteout = cv2.resize(img, self.resize_shape)
            img_whiteout = 1 - img_whiteout / 255

            tmp = np.sum(img_block, axis=2)
            # print(len(tmp[tmp > 0]))
            # print(len(tmp[tmp == 0]))
            for i in range(img_block.shape[0]):
                for j in range(img_block.shape[1]):
                    if tmp[i, j] > 0.1:
                        img_whiteout[i, j, :] = img_block[i, j, :]
            res.append(img_whiteout)
        return np.asarray(res)


if __name__ == "__main__":
    data_parser = DataParserV2('../dataset', (512, 512), list_files=['../dataset/image_list_1.txt'], batch_size=1)
    condition = data_parser.get_batch_condition_add()
    raw = data_parser.get_batch_raw()
    raw = (1 - raw) * 255
    raw = cv2.cvtColor(raw[0], cv2.COLOR_BGR2RGB)

    whiteout = data_parser.get_batch_color_hint_whiteout()
    whiteout = (1 - whiteout) * 255
    whiteout = cv2.cvtColor(whiteout[0], cv2.COLOR_BGR2RGB)

    img = (1 - condition) * 255
    img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(raw.astype(np.uint8))
    plt.subplot(1, 3, 2)
    plt.imshow(whiteout.astype(np.uint8))
    plt.subplot(1, 3, 3)
    plt.imshow(img.astype(np.uint8))
    plt.show()
