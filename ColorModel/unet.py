import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataProcess.data_parser import DataParser
import os
import cv2


# base line
class Unet:
    def __init__(self, image_shape, batch_size):
        self.input = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 2), name='input')
        self.condition = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 6),
                                        name='condition')
        self.target = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 3), name='output')
        self.batch_size = batch_size

    def generator(self):
        inputs = tf.concat(values=[self.input, self.condition], axis=3)
        with tf.variable_scope('generator'):
            encode1, sample_encode1 = self.cnn_block(inputs, 64, (3, 3), sample_type='conv', scope_name='encode1')
            encode2, sample_encode2 = self.cnn_block(sample_encode1, 128, (3, 3), sample_type='conv',
                                                     scope_name='encode2')
            encode3, sample_encode3 = self.cnn_block(sample_encode2, 256, (3, 3), sample_type='conv',
                                                     scope_name='encode3')
            encode4, sample_encode4 = self.cnn_block(sample_encode3, 512, (3, 3), sample_type='conv',
                                                     scope_name='encode4')

            with tf.variable_scope('last_encode'):
                layer = tf.layers.conv2d(inputs=sample_encode4, filters=1024, kernel_size=(3, 3),
                                         activation=tf.nn.leaky_relu, name='conv1', padding='same')
                layer = tf.layers.conv2d(inputs=layer, filters=1024, kernel_size=(3, 3),
                                         activation=tf.nn.leaky_relu, name='conv2', padding='same')
            decode1, _ = self.cnn_block(layer, 512, (3, 3), sample_type='deconv', scope_name='decode1',
                                        deconv_concatenate=encode4)
            decode2, _ = self.cnn_block(decode1, 256, (3, 3), sample_type='deconv', scope_name='decode2',
                                        deconv_concatenate=encode3)
            decode3, _ = self.cnn_block(decode2, 128, (3, 3), sample_type='deconv', scope_name='decode3',
                                        deconv_concatenate=encode2)
            decode4, _ = self.cnn_block(decode3, 64, (3, 3), sample_type='deconv', scope_name='decode4',
                                        deconv_concatenate=encode1)

            g_logits = tf.layers.conv2d(decode4, 3, (3, 3), padding='same', name='logits')

            # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=g_logits)
            # cost = tf.reduce_mean(loss)
            # optimizer = tf.train.AdamOptimizer().minimize(cost)
            return g_logits

    def discriminator(self, input, reuse=False):
        inputs = tf.concat(values=[input, self.condition], axis=3)
        with tf.variable_scope('discriminator', reuse=reuse):
            layer = tf.layers.conv2d(inputs, 64, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv1')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool1')
            layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv2')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool2')
            layer = tf.layers.conv2d(layer, 256, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv3')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool3')
            layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv4')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool4')
            layer = tf.layers.flatten(layer)
            d_logits = tf.layers.dense(layer, 1000, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       activation=tf.nn.relu)
            d_logits = tf.layers.dense(d_logits, 10,
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            return d_logits

    def cnn_block(self, input, num_filter, kernel_size, sample_type, scope_name, deconv_concatenate=None, reuse=False):
        if sample_type not in ['conv', 'deconv']:
            raise ValueError('Undefined sample type')
        with tf.variable_scope(scope_name, reuse=reuse):
            if sample_type == 'conv':
                cnn1 = tf.layers.conv2d(inputs=input, filters=num_filter, kernel_size=kernel_size, padding='same',
                                        activation=tf.nn.leaky_relu, name='conv1',
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                out = tf.layers.conv2d(inputs=cnn1, filters=num_filter, kernel_size=kernel_size, padding='same',
                                       activation=tf.nn.leaky_relu, name='conv2',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                sample_out = tf.layers.max_pooling2d(out, pool_size=(2, 2), strides=(2, 2), name='maxpool')

            else:
                if deconv_concatenate is None:
                    raise ValueError('deconv_concatenate can not be None when building deconv structurej')
                dcnn = tf.layers.conv2d_transpose(inputs=input, filters=num_filter, kernel_size=kernel_size, strides=2,
                                                  padding='same', activation=tf.nn.leaky_relu, name='deconv1',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                concat = tf.concat(values=[dcnn, deconv_concatenate], axis=3)
                dcnn1 = tf.layers.conv2d(inputs=concat, filters=num_filter, kernel_size=kernel_size, padding='same',
                                         activation=tf.nn.leaky_relu, name='d_conv1',
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                out = tf.layers.conv2d(inputs=dcnn1, filters=num_filter, kernel_size=kernel_size, padding='same',
                                       activation=tf.nn.leaky_relu, name='d_conv2',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                sample_out = None
            return out, sample_out

    def train(self, inputs, targets, conditions, epochs, learning_rate=0.01, clip_low=-1, clip_high=1):
        d_input = tf.placeholder(tf.float32, shape=(None, inputs.shape[1], inputs.shape[2], 3), name='d_inputs')
        g_logits = self.generator()
        d_logits_real = self.discriminator(d_input, reuse=False)
        d_logits_fake = self.discriminator(g_logits, reuse=True)

        g_loss = tf.reduce_mean(d_logits_fake)
        d_loss = tf.reduce_mean(d_logits_real - d_logits_fake)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_loss', d_loss)

        var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        var_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

        d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=var_d)
        g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=var_g)

        clip_d_var = [var.assign(tf.clip_by_value(var, clip_low, clip_high)) for var in var_d]

        data_parser_inputs = DataParser(inputs, batch_size=self.batch_size)
        data_parser_targets = DataParser(targets, batch_size=self.batch_size)
        data_parser_conditions = DataParser(conditions, batch_size=self.batch_size)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./logs/train', sess.graph)
            writer_val = tf.summary.FileWriter('./logs/val')
            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                loss_g = 0
                loss_d = 0
                for i in range(data_parser_inputs.iteration):
                    batch_input = data_parser_inputs.get_batch()
                    batch_target = data_parser_targets.get_batch()
                    batch_condition = data_parser_conditions.get_batch()

                    _, loss_g = sess.run([g_optimizer, g_loss],
                                         feed_dict={self.input: batch_input, self.condition: batch_condition})
                    _, loss_d = sess.run([d_optimizer, d_loss],
                                         feed_dict={d_input: batch_target, self.input: batch_input,
                                                    self.condition: batch_condition})
                    sess.run(clip_d_var)

                    log = sess.run(merged)
                    writer.add_summary(log, epoch * data_parser_inputs.iteration + i + 1)

                    if i % 10 == 0:
                        log_val, val_loss_g, val_loss_d = sess.run([merged, g_loss, d_loss],
                                                                   feed_dict={self.input: batch_input,
                                                                              self.condition: batch_condition,
                                                                              d_input: batch_target})
                        writer_val.add_summary(log_val, epoch * data_parser_inputs.iteration + i + 1)
                        print(
                            'Epoch: {}, Iteration: {}, g_loss: {}, d_loss: {}, val_g_loss: {}, val_d_loss: {}'.format(
                                epoch + 1, i + 1, loss_g, loss_d, val_loss_g, val_loss_d))
                print('*' * 10,
                      'Epoch {}/{} ...'.format(epoch + 1, epochs),
                      'd_loss: {:.4f} ...'.format(loss_d),
                      'g_loss: {:.4f} ...'.format(loss_g),
                      '*' * 10)
                saver.save(sess, './checkpoints/checkpoint_{}.ckpt'.format(epoch + 1))

            saver.save(sess, './checkpoints/model.ckpt')


if __name__ == '__main__':
    dir_num = 1
    base_path = '../dataset/raw_image'  # where the raw images locate
    sketch_output = '../dataset/sketch'  # directory to store sketches
    color_hint_output = '../dataset/color_hint'  # directory to store color hint(blur)
    color_hint_whiteout_output = '../dataset/color_hint_with_whiteout'  # directory to store color hint added whiteout
    color_block_output = '../dataset/color_block'  # directory to store color block
    resize_shape = (512, 512)  # raw images should be resized

    list_file = '../dataset/image_list_{:d}.txt'.format(dir_num)  # get image list
    with open(list_file, 'r') as f:
        images_path = f.readlines()
    targets = []
    inputs = []
    conditions = []
    for count, path in enumerate(images_path):
        path = path.strip('\n')
        img_path = os.path.join(base_path, path)
        img_name = path.split('/')[-1]
        img_num = img_name.split('/')[-1].split('.')[0]
        sketch_name = img_num + '_sketch.jpg'
        color_hint_name = img_num + '_colorhint.jpg'
        color_hint_whiteout_name = img_num + '_whiteout.jpg'
        color_block_name = img_num + '_colorblock.jpg'

        target = cv2.imread(img_path)
        target = cv2.resize(target, resize_shape)
        targets.append(target / 255)

        sketch = cv2.imread(os.path.join(os.path.join(sketch_output, '{:03d}'.format(dir_num)), sketch_name), 0) / 255
        noises = np.random.rand(1000)
        indices = np.random.permutation(resize_shape[0] * resize_shape[1])[:1000]
        noise_channel = np.zeros_like(sketch)
        for i, val in enumerate(indices):
            noise_channel[val // resize_shape[0]][val % resize_shape[1]] = noises[i]
        sketch = np.expand_dims(sketch, axis=2)
        noise_channel = np.expand_dims(noise_channel, axis=2)
        sketch_noise = np.concatenate((sketch, noise_channel), axis=2)
        inputs.append(sketch_noise)
        # sketch = np.expand_dims(sketch, axis=2)

        # color_hint = cv2.imread(
        #     os.path.join(os.path.join(color_hint_output, '{:03d}'.format(dir_num)), color_hint_name)) / 255
        color_hint_whiteout = cv2.imread(
            os.path.join(os.path.join(color_hint_whiteout_output, '{:03d}'.format(dir_num)),
                         color_hint_whiteout_name)) / 255
        color_block = cv2.imread(
            os.path.join(os.path.join(color_block_output, '{:03d}'.format(dir_num)), color_block_name)) / 255
        condition = np.concatenate((color_hint_whiteout, color_block), axis=2)
        print(count+1)
        if count >= 100:
            break
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    conditions = np.asarray(conditions)

    model = Unet(resize_shape, batch_size=128)
    model.train(inputs, targets, conditions, 10)
