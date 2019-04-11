import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataProcess.data_parser import *
import os
import cv2


# base line
class Unet:
    def __init__(self, image_shape, batch_size):
        self.image_shape = image_shape
        self.batch_size = batch_size

    def generator(self, input, condition):
        inputs = tf.concat(values=[input, condition], axis=3)
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

    def discriminator(self, input, condition, reuse=False):
        inputs = tf.concat(values=[input, condition], axis=3)
        with tf.variable_scope('discriminator', reuse=reuse):
            layer = tf.layers.conv2d(inputs, 64, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv1')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool1')
            layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv2')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool2')
            layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv3')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool3')
            layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.leaky_relu, name='conv4')
            layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='maxpool4')
            layer = tf.layers.flatten(layer)
            d_logits = tf.layers.dense(layer, 1000, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       activation=tf.nn.relu)
            d_logits = tf.layers.dense(d_logits, 1,
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
                    raise ValueError('deconv_concatenate can not be None when building deconv structure')
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

    def train(self, dataset_path, list_files, epochs, learning_rate=0.01, clip_low=-0.01, clip_high=0.01, r=0.01):
        tf.reset_default_graph()
        input = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 1), name='input')
        condition = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3),
                                   name='condition')
        target = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3), name='output')
        g_logits = self.generator(input, condition)
        d_logits_real = self.discriminator(target, condition, reuse=False)
        d_logits_fake = self.discriminator(g_logits, condition, reuse=True)

        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(g_logits - input), axis=0))
        g_loss = -tf.reduce_mean(d_logits_fake) + r * l1_loss
        d_loss = -tf.reduce_mean(tf.abs(d_logits_fake - d_logits_real))
        # tf.summary.scalar('g_loss', g_loss)
        # tf.summary.scalar('d_loss', d_loss)

        var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        var_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

        d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=var_d)
        g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=var_g)

        clip_d_var = [var.assign(tf.clip_by_value(var, clip_low, clip_high)) for var in var_d]

        data_parser = DataParserV2(dataset_path, self.image_shape, list_files=list_files, batch_size=self.batch_size)
        data_parser_test = DataParserV2(dataset_path, self.image_shape, list_files=list_files, batch_size=5)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        losses = []
        test_input = data_parser_test.get_batch_sketch()
        test_target = data_parser_test.get_batch_raw()
        test_condition = data_parser_test.get_batch_condition_add()
        test = {'input': test_input, 'target': test_target, 'condition': test_condition}
        np.save('./results/test_data_3.npy', test)
        outputs = []
        with tf.Session(config=config) as sess:
            # print('enter')
            # writer = tf.summary.FileWriter('./logs/train', sess.graph)
            # writer_val = tf.summary.FileWriter('./logs/val')
            saver = tf.train.Saver()
            # merged = tf.summary.merge_all()
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            sess.run(tf.global_variables_initializer())
            sess.graph.finalize()
            # print('initializing')
            for epoch in range(epochs):
                loss_g = 0
                loss_d = 0
                for i in range(data_parser.iteration):
                    batch_input = data_parser.get_batch_sketch()
                    # print(batch_input.dtype, batch_input.shape)
                    batch_target = data_parser.get_batch_raw()
                    # print(batch_target.dtype, batch_target.shape)
                    batch_condition = data_parser.get_batch_condition_add()
                    # print(batch_condition.dtype, batch_condition.shape)
                    data_parser.update_iterator()

                    for _ in range(5):
                        _, loss_d = sess.run([d_optimizer, d_loss],
                                             feed_dict={target: batch_target, input: batch_input,
                                                        condition: batch_condition}, options=run_options)
                        sess.run(clip_d_var)
                    # print('discriminator')
                    # print('clip')
                    _, loss_g, loss_l1 = sess.run([g_optimizer, g_loss, l1_loss],
                                                  feed_dict={input: batch_input, condition: batch_condition},
                                                  options=run_options)
                    # print('generator')

                    # log = sess.run(merged)
                    # writer.add_summary(log, epoch * data_parser_inputs.iteration + i + 1)
                    losses.append([loss_d, loss_g, loss_l1])
                    if i % 10 == 0:
                        # log_val, val_loss_g, val_loss_d = sess.run([merged, g_loss, d_loss],
                        #                                            feed_dict={input: batch_input,
                        #                                                       condition: batch_condition,
                        #                                                       target: batch_target})
                        # writer_val.add_summary(log_val, epoch * data_parser_inputs.iteration + i + 1)
                        print(
                            'Epoch: {}, Iteration: {}/{}, g_loss: {}, d_loss: {}, l1 loss: {}'.format(
                                epoch + 1, i + 1, data_parser.iteration, loss_g, loss_d, loss_l1))
                print('*' * 10,
                      'Epoch {}/{} ...'.format(epoch + 1, epochs),
                      'd_loss: {:.4f} ...'.format(loss_d),
                      'g_loss: {:.4f} ...'.format(loss_g),
                      '*' * 10)
                output = sess.run(g_logits, feed_dict={input: test_input, condition: test_condition})
                outputs.append(output)
                saver.save(sess, './checkpoints/unet3/checkpoint_{}.ckpt'.format(epoch + 1))

            saver.save(sess, './checkpoints/unet3/model.ckpt')
            np.save('./results/predicted_3.npy', outputs)
        return losses


if __name__ == '__main__':
    dataset_path = '../dataset'
    resize_shape = (256, 256)
    list_files = ['../dataset/image_list_1.txt']
    batch_size = 16
    # data_parser = DataParserV2(dataset_path, resize_shape, list_files, batch_size)

    model = Unet(resize_shape, batch_size=batch_size)
    losses = model.train(dataset_path, list_files, 10, learning_rate=0.0005)
    np.save('./logs/train/losses_2.npy', np.asarray(losses))

    losses = np.asarray(losses)
    plt.plot(losses[:, 0])
    plt.title('d loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(losses[:, 1])
    plt.title('g loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(losses[:, 2])
    plt.title('l1 loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    imgs = np.load('./results/predicted_3.npy')
    test = np.load('./results/test_data_3.npy')
    img = test.item()['target'][0]
    img = (1 - img) * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    plt.figure()
    for i in range(imgs.shape[0]):
        img = imgs[i, 0]
        img = (1 - img) * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
    plt.show()
