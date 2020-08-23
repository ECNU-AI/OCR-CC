import tensorflow as tf
from PIL import Image
import numpy as np
import born_data
import cv2
import sys


class ChineseCodeRecognition():
    """docstring for ChineseCodeRecognition"""

    def __init__(self):
        self.w3500 = open('3500.txt', 'r').read()
        self.x_1, self.sess_1, self.if_is_training_1, self.y_loca = self.load_model_1()
        self.x_2, self.sess_2, self.if_is_training_2, self.y_class = self.load_model_2()

    def load_model_1(self):
        graph_1 = tf.Graph()
        sess_1 = tf.Session(graph=graph_1)
        with graph_1.as_default():
            model_saver_1 = tf.train.import_meta_graph("./model_step1/mymodel.ckpt.meta")

            model_saver_1.restore(sess_1, './model_step1/mymodel.ckpt')
            y_loca = tf.get_collection('pre_loca')[0]
            x_1 = graph_1.get_operation_by_name('in_image').outputs[0]
            if_is_training_1 = graph_1.get_operation_by_name('if_is_training').outputs[0]

            return x_1, sess_1, if_is_training_1, y_loca

    def load_model_2(self):
        graph_2 = tf.Graph()
        sess_2 = tf.Session(graph=graph_2)
        with graph_2.as_default():
            model_saver_2 = tf.train.import_meta_graph("./model_step2/mymodel.ckpt.meta")
            model_saver_2.restore(sess_2, './model_step2/mymodel.ckpt')
            y_class = tf.get_collection('pre_img')[0]

            x_2 = graph_2.get_operation_by_name('in_image').outputs[0]
            if_is_training_2 = graph_2.get_operation_by_name('if_is_training').outputs[0]

            return x_2, sess_2, if_is_training_2, y_class

    def readImage(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, data = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        data = data / 255
        return data

    def crop_image(self, data, loca_np, imgshow=False):

        croped_img_list = []

        loca_list = loca_np.tolist()
        if imgshow:
            img = data.copy()
        m, n = loca_np.shape
        for i in range(m):
            x = round(loca_list[i][0] * 100 - 10)
            y = round(loca_list[i][1] * 25 - 10)
            if x < 0:
                x = 0
            elif x > 80:
                x = 80
            if y < 0:
                y = 0
            elif y > 9:
                y = 9
            temp = data[y:y + 20, x:x + 20]
            croped_img_list.append(temp.tolist())
            if imgshow:
                img = cv2.rectangle(img * 255, (x, y), (x + 20, y + 20), (255, 0, 0), 1)
        if imgshow:
            img = Image.fromarray(img)
            img.show()
        # 返回的是0～1的图片，类型List
        return croped_img_list

    # 预测单张验证码
    def predict(self):
        while True:
            try:
                filename = input('input image filename:')
                if filename == 'qq':
                    break
                data = self.readImage(filename)
            except Exception as e:
                print('please check filename')
                continue

            in_image = data.reshape(1, 30, 100)
            loca_np = self.sess_1.run(self.y_loca, feed_dict={self.x_1: in_image, self.if_is_training_1: False})
            loca_np = loca_np.reshape(4, 2)
            imgCutList = self.crop_image(data, loca_np, True)
            chineseCode = ""
            for imList in imgCutList:
                data_2 = np.array(imList).reshape(1, 400)

                # data=tf.reshape(data, shape=[1,400])
                rel = self.sess_2.run(self.y_class, feed_dict={self.x_2: data_2, self.if_is_training_2: False})
                num = np.argmax(rel)
                chineseCode += self.w3500[num]
            print(chineseCode)

    # 测试准确率，测试times张验证码
    def test(self, times):
        erro = 0
        loss = 0
        for i in range(times):
            i_chr = born_data.ImageChar()
            img_PIL, words = i_chr.rand_img_test()

            in_img = np.asarray(img_PIL)
            in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
            ret, in_img = cv2.threshold(in_img, 127, 255, cv2.THRESH_BINARY_INV)
            data = in_img / 255
            in_image = data.reshape(1, 30, 100)

            loca_np = self.sess_1.run(self.y_loca, feed_dict={self.x_1: in_image, self.if_is_training_1: False})
            loca_np = loca_np.reshape(4, 2)
            imgCutList = self.crop_image(data, loca_np)
            chineseCode = ""
            for imList in imgCutList:
                try:
                    data_2 = np.array(imList).reshape(1, 400)
                except Exception as e:
                    loss += 1
                    continue
                rel = self.sess_2.run(self.y_class, feed_dict={self.x_2: data_2, self.if_is_training_2: False})
                num = np.argmax(rel)
                chineseCode += self.w3500[num]
            if len(chineseCode) == 4:
                if not chineseCode == words:
                    erro += 1
            print('\r', i, end='\r')

        print('erro: ', erro / times * 100, '%', '\tloss: ', loss)


if __name__ == '__main__':
    ccr = ChineseCodeRecognition()
    ccr.test(10000)
# ccr.predict()
