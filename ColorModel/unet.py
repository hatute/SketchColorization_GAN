import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Unet:
    def __init__(self, image_shape, batch_size):
        self.input = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 1), name='input')
        self.target = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 3), name='output')
        self.batch_size = batch_size

    def generator(self):
        with tf.variable_scope('generator'):
            encode1, sample_encode1 = self.cnn_block(self.input, 64, (3, 3), sample_type='conv', scope_name='encode1')
            encode2, sample_encode2 = self.cnn_block(sample_encode1, 128, (3, 3), sample_type='conv',
                                                     scope_name='encode2')
            encode3, sample_encode3 = self.cnn_block(sample_encode2, 256, (3, 3), sample_type='conv',
                                                     scope_name='encode3')
            encode4, sample_encode4 = self.cnn_block(sample_encode3, 512, (3, 3), sample_type='conv',
                                                     scope_name='encode4')

            with tf.variable_scope('last_encode'):
                layer = tf.layers.conv2d(inputs=sample_encode4, filters=1024, kernel_size=(3, 3),
                                         activation=tf.nn.leaky_relu, name='conv1')
                layer = tf.layers.conv2d(inputs=layer, filters=1024, kernel_size=(3, 3),
                                         activation=tf.nn.leaky_relu, name='conv2')
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
        with tf.variable_scope('discriminator', reuse=reuse):
            layer = tf.layers.conv2d(input, 64, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv1')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool1')
            layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv2')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool2')
            layer = tf.layers.conv2d(layer, 256, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv3')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool3')
            layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv4')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool4')
            layer = tf.layers.flatten(layer)
            d_logits = tf.layers.dense(layer, 1000, use_bias=False,
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


if __name__ == '__main__':
    pass
