import tensorflow.contrib.slim as slim
import tensorflow as tf
import born_data
from PIL import Image
import numpy as np
import random
import time
import cv2


def generate_data():
    img_char = born_data.ImageChar()
    images = []
    labels = []
    for i in range(30):
        chinese_img_PIL, label_list = img_char.rand_img_label(step=2)
        np_img = np.asarray(chinese_img_PIL)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        ret, np_img = cv2.threshold(np_img, 127, 255, cv2.THRESH_BINARY_INV)
        np_img = np_img / 255
        images.append(np_img.tolist())
        labels.append(label_list)

    return np.array(images), np.array(labels)


def crop_image(data, loca_np, imgshow=False):
    croped_img_list = []

    loca_list = loca_np.tolist()
    if imgshow:
        img = data.copy()
    m, n = loca_np.shape
    for i in range(m):
        x = round(loca_list[i][0] * 100 - 10)  # 将中心横坐标转化为左上角横坐标，方便剪裁
        y = round(loca_list[i][1] * 25 - 10)  # 将中心纵坐标转化为左上角纵坐标
        # 根据坐标剪裁可能会超出边界。
        if x < 0:
            x = 0
        elif x > 80:
            x = 80
        if y < 0:
            y = 0
        elif y > 9:
            y = 9

        temp = data[y:y + 20, x:x + 20]  # 对汉字进行剪裁
        croped_img_list.append(temp.tolist())
        if imgshow:
            img = cv2.rectangle(img * 255, (x, y), (x + 20, y + 20), (255, 0, 0), 1)
    if imgshow:
        img = Image.fromarray(img)
        img.show()
    # 返回的是0～1的图片，类型List
    return croped_img_list


def cal_loss(y_pre, y_label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_pre))
    # return -tf.reduce_sum(y_label*tf.log(y_pre))
    # return tf.reduce_mean(tf.square(y_label - y_pre))
    # return tf.reduce_mean(tf.pow(tf.subtract(y_pre,y_label),2))


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
        net = slim.fully_connected(net, 3000,
                                   weights_initializer=slim.xavier_initializer(),
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='fc1')
        print('fc1:\t', net.get_shape())

        net = slim.fully_connected(net, 9000,
                                   weights_initializer=slim.xavier_initializer(),
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='fc2')
        print('fc2:\t', net.get_shape())

        net = slim.fully_connected(net, 3500,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   # weights_initializer=slim.xavier_initializer(),
                                   # biases_initializer=tf.zeros_initializer(),
                                   scope='fc3')
        print('soft:\t', net.get_shape())

        return net


def accuracy(sess, pre_image, in_image, testImg, testLab, if_is_training):
    erro_count = 0

    for i in range(10):
        bt = random.randint(0, 4999 - 100)
        x_image_2 = testImg[bt:bt + 100, :]
        y_label_2 = testLab[bt:bt + 100, :]
        pre_label = sess.run(pre_image, feed_dict={in_image: x_image_2, if_is_training: False})
        M, N = pre_label.shape
        for m in range(M):
            x = np.argmax(pre_label[m, :])
            x0 = np.argmax(y_label_2[m, :])
            if not x == x0:
                erro_count += 1
    if erro_count <= 2:
        return True, erro_count
    else:
        return False, erro_count


# 加载第一个网络的模型。
def load_model_1():
    graph_1 = tf.Graph()
    sess_1 = tf.Session(graph=graph_1)
    with graph_1.as_default():
        model_saver_1 = tf.train.import_meta_graph("./model_step1/mymodel.ckpt.meta")

        model_saver_1.restore(sess_1, './model_step1/mymodel.ckpt')
        y_loca = tf.get_collection('pre_loca')[0]
        x_1 = graph_1.get_operation_by_name('in_image').outputs[0]
        if_is_training_1 = graph_1.get_operation_by_name('if_is_training').outputs[0]

        return x_1, sess_1, if_is_training_1, y_loca


# 第一个模型对输入数据进行预测中心坐标
def pre_model_1(x_1, sess_1, if_is_training_1, y_loca, in_image_1, y_label):
    loca_np = sess_1.run(y_loca, feed_dict={x_1: in_image_1, if_is_training_1: False})
    M, N, L = loca_np.shape
    x_image_2 = []
    y_label_2 = []
    for m in range(M):
        imgCutList = crop_image(in_image_1[m, :, :], loca_np[m, :, :])
        for im in range(len(imgCutList)):
            try:
                data_2 = np.array(imgCutList[im]).reshape(400, )
            except Exception as e:
                print('imList reshape erro')
                continue
            x_image_2.append(data_2.tolist())
            y_label_2.append(y_label[m, im, :].tolist())

    return np.array(x_image_2), np.array(y_label_2)


def main():
    in_image = tf.placeholder(dtype=tf.float32, shape=[None, 400], name='in_image')
    out_image = tf.placeholder(dtype=tf.float32, shape=[None, 3500], name='out_image')

    # 和 batch normalization一起使用，在训练时为True，预测时False
    if_is_training = tf.placeholder(dtype=tf.bool, name='if_is_training')

    x_input = tf.reshape(in_image, shape=[-1, 20, 20, 1], name='x_input')

    pre_image = network(x_input, if_is_training)

    # l2_loss = tf.add_n(tf.losses.get_regularization_losses())
    cost = cal_loss(pre_image, out_image)
    corr = tf.equal(tf.argmax(pre_image, 1), tf.argmax(out_image, 1))
    loss = tf.reduce_mean(tf.cast(corr, "float"))

    # 和 batch normalization 一起使用
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(cost)

    model_saver = tf.train.Saver()
    tf.add_to_collection('pre_img', pre_image)

    testImg = np.load('testImg1.npy')
    testLab = np.load('testLab1.npy')

    x_1, sess_1, if_is_training_1, y_loca = load_model_1()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            # 输入训练次数，方便控制和继续训练
            command = input('input:\t')
            if command == 'qq':
                break
            for i in range(int(command)):
                x_image_1, y_label = generate_data()
                x_image_2, y_label_2 = pre_model_1(x_1, sess_1, if_is_training_1, y_loca, x_image_1, y_label)
                sess.run(train_op, feed_dict={in_image: x_image_2, out_image: y_label_2, if_is_training: True})

                if i % 500 == 0:
                    ret, erro_count = accuracy(sess, pre_image, in_image, testImg, testLab, if_is_training)
                    print('count: ', i, '\taccuracy: ', erro_count)

        model_saver.save(sess, './model_step2/mymodel.ckpt')


if __name__ == '__main__':
    main()

# x_1 , sess_1 , if_is_training_1 ,y_loca =load_model_1()
# x_image_1,y_label = generate_data()
# x_image_2,y_label_2= pre_model_1(x_1 , sess_1 , if_is_training_1 ,y_loca,x_image_1,y_label)
# np.save('trainImg2',x_image_2)
# np.save('trainLab2',y_label_2)