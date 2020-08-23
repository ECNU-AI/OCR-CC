import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import random
import time


def cal_loss(pre_loca, lab_loca):
    loca_loss = tf.reduce_mean(tf.square(tf.subtract(pre_loca, lab_loca)))
    return loca_loss * 100


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def network(in_image, if_is_training):
    batch_norm_params = {
        'is_training': if_is_training,
        'zero_debias_moving_mean': True,
        'decay': 0.99,
        'epsilon': 0.001,
        'scale': True,
        'updates_collections': None
    }

    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        padding='SAME',
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        out_1 = 32
        out_2 = 64
        out_3 = 128

        net = slim.conv2d(in_image, num_outputs=out_2, kernel_size=[5, 5], stride=1, scope='conv1')
        print('1_con:\t', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1')
        print('1_pool:\t', net.get_shape())

        net = slim.conv2d(net, num_outputs=out_2, kernel_size=[5, 5], stride=1, scope='conv2')
        print('2_con:\t', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool2')
        print('2_pool:\t', net.get_shape())

        net = slim.conv2d(net, num_outputs=out_3, kernel_size=[3, 3], stride=1, scope='conv3_1')
        net = slim.conv2d(net, num_outputs=out_3, kernel_size=[3, 3], stride=1, scope='conv3_2')
        print('3_con:\t', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool3')
        print('3_pool:\t', net.get_shape())

    # net = tf.reshape(net,shape=[-1,2*2*128])
    net = slim.flatten(net, scope='flatten')

    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        net = slim.fully_connected(net, 1000,
                                   weights_initializer=slim.xavier_initializer(),
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='fc_total')
        print('fc:\t', net.get_shape())

        pre_loca = slim.fully_connected(net, 2000,
                                        weights_initializer=slim.xavier_initializer(),
                                        biases_initializer=tf.zeros_initializer(),
                                        scope='fc2_1')

        pre_loca = slim.fully_connected(pre_loca, 8,
                                        activation_fn=tf.nn.sigmoid,
                                        # normalizer_fn=None,
                                        weights_initializer=slim.xavier_initializer(),
                                        biases_initializer=tf.zeros_initializer(),
                                        scope='fc2_2')

        pre_loca = tf.reshape(pre_loca, shape=[-1, 4, 2])
        return pre_loca


# 测试网络训练精度：预测点坐标和标签点坐标相差两个像素以上视为预测失败。
def accuracy(sess, pre_loca, in_image, x_image, y_label, if_is_training):
    erro_count = 0
    for i in range(10):  # 每次取一百张预测，取十次共1000
        bt = random.randint(0, 49999 - 100)
        min_x_image = x_image[bt:(bt + 100), :, :]
        min_y_label = y_label[bt:(bt + 100), :, :]
        loca_np = sess.run(pre_loca, feed_dict={in_image: min_x_image, if_is_training: True})
        m, n, l = loca_np.shape
        for j in range(m):
            for k in range(n):
                x = round(loca_np[j, k, 0] * 100)
                y = round(loca_np[j, k, 1] * 25)
                x0 = round(min_y_label[j, k, 0] * 100)
                y0 = round(min_y_label[j, k, 1] * 25)
                lo = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5  # 计算两个预测坐标和标签坐标的距离
                if lo > 2:
                    erro_count += 1
    if erro_count > 20:
        return False, erro_count
    else:
        return True, erro_count


def main():
    in_image = tf.placeholder(dtype=tf.float32, shape=[None, 30, 100], name='in_image')
    # lab_class=tf.placeholder(dtype=tf.float32, shape=[None,4,3500], name='lab_class')
    lab_loca = tf.placeholder(dtype=tf.float32, shape=[None, 4, 2], name='lab_loca')

    # 和 batch normalization一起使用，在训练时为True，预测时False
    if_is_training = tf.placeholder(dtype=tf.bool, name='if_is_training')

    x_input = tf.reshape(in_image, shape=[-1, 100, 30, 1], name='x_input')

    pre_loca = network(x_input, if_is_training)

    loca_loss = cal_loss(pre_loca, lab_loca)

    # 和 batch normalization 一起使用
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(loca_loss)

    model_saver = tf.train.Saver()
    tf.add_to_collection('pre_loca', pre_loca)

    x_image = np.load('trainImg0.npy')
    y_label = np.load('trainLab0.npy')

    batchs = 120
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            # 输入训练次数，方便控制和继续训练
            command = input('input:\t')
            if command == 'qq':
                break
            for i in range(int(command)):
                bt = random.randint(0, 49999 - batchs)
                min_x_image = x_image[bt:(bt + batchs), :, :]
                min_y_label = y_label[bt:(bt + batchs), :, :]

                sess.run(train_op, feed_dict={in_image: min_x_image, lab_loca: min_y_label, if_is_training: True})

                if i % 500 == 0:
                    ret, erro_count = accuracy(sess, pre_loca, in_image, x_image, y_label, if_is_training)
                    print('count: ', i, '\terro: ', erro_count, '\t\taccuracy: ', erro_count / 4000)
                    if ret:
                        break

        model_saver.save(sess, './model/mymodel.ckpt')


if __name__ == '__main__':
    main()
