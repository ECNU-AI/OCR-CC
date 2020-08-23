from PIL import Image, ImageFont, ImageDraw
import random
import os
import numpy as np
import cv2


class ImageChar():
    """
    1、读取3500.txt 这是最常用3500汉字 并随机挑选出汉字
    2、在./fonts/ 文件夹下存放 字体格式 随机挑选格式 然后依据格式随机生成汉字
    3、随机画指定数目的干扰线
    4、环境：Mac python3.5
    """

    def __init__(self, color=(0, 0, 0), size=(100, 30),
                 fontlist=['./fonts/' + i for i in os.listdir('./fonts/') if not i == '.DS_Store'],
                 fontsize=20,
                 num_word=4):  # 生成多少个字的验证码（图片宽度会随之增加）

        self.num_word = num_word
        self.color = color

        self.fontlist = fontlist

        if self.num_word == 4:
            self.size = size
        else:
            self.size = ((self.fontsize + 5) * self.num_word, 40)

        # 随机挑选一个字体 randint(0,2)会取0，1，2 所以减去 1
        self.fontpath = self.fontlist[random.randint(0, len(self.fontlist) - 1)]
        self.fontsize = fontsize

        self.chinese = open('3500.txt', 'r').read()

        self.font = ImageFont.truetype(self.fontpath, self.fontsize)

    # 随机生成四个汉字的字符串
    def rand_chinese(self):
        chinese_str = ''
        chinese_num = []
        for i in range(self.num_word):
            temp = random.randint(0, 10)
            chinese_str = chinese_str + self.chinese[temp]
            chinese_num.append(temp)
        return chinese_str, chinese_num

    # 随机生成杂线的坐标
    def rand_line_points(self, mode=0):
        width, height = self.size
        if mode == 0:
            return (random.randint(0, width), random.randint(0, height))
        elif mode == 1:
            return (random.randint(0, 6), random.randint(0, height))
        elif mode == 2:
            return (random.randint(width - 6, width), random.randint(0, height))

    # 随机生成一张验证码，并且返回 四个汉字的字符串，测试用
    def rand_img_test(self, num_lines=4):
        width, height = self.size
        gap = 5
        start = 0

        # 第一张，带噪音的验证码
        self.img1 = Image.new('RGB', self.size, (255, 255, 255))
        self.draw1 = ImageDraw.Draw(self.img1)

        # 把线画上去
        for i in range(num_lines // 2):
            self.draw1.line([self.rand_line_points(), self.rand_line_points()], (0, 0, 0))
        for i in range(num_lines // 2):
            self.draw1.line([self.rand_line_points(1), self.rand_line_points(2)], (0, 0, 0))

        words, chinese_num = self.rand_chinese()
        # 将汉字画上去
        for i in range(len(words)):
            x = start + (self.fontsize + gap) * i + random.randint(0, gap)
            y = random.randint(0, height - self.fontsize - gap)
            self.draw1.text((x, y), words[i], fill=(0, 0, 0), font=self.font)
        return self.img1, words

    # 随机生成一张图片 根据step值，分别为第一个网络和第二个网络提供训练数据
    def rand_img_label(self, num_lines=4, step=1):
        width, height = self.size
        gap = 5
        start = 0

        self.img1 = Image.new('RGB', self.size, (255, 255, 255))
        self.draw1 = ImageDraw.Draw(self.img1)

        # 把线画上去
        for i in range(num_lines // 2):
            self.draw1.line([self.rand_line_points(), self.rand_line_points()], (0, 0, 0))
        for i in range(num_lines // 2):
            self.draw1.line([self.rand_line_points(1), self.rand_line_points(2)], (0, 0, 0))

        words, chinese_num = self.rand_chinese()
        label_list = []
        # 将汉字画上去
        for i in range(len(words)):
            x = start + (self.fontsize + gap) * i + random.randint(0, gap)
            y = random.randint(0, height - self.fontsize - gap)

            if step == 1:  # 为第一个网络生成标签数据：汉字的坐标
                temp_list = [0] * 2
                temp_list[0] = (x + 10) / 100  # 该汉字的中心横坐标，除于100是为了规划到0～1，为了方便训练
                temp_list[1] = (y + 14) / 25  # 该汉字的中心纵坐标，除于25也是为了方便训练
            else:  # 为第二个网络生成标签数据，汉字的one-hot矩阵
                temp_list = [0] * 3500
                temp_list[chinese_num[i]] = 1

            label_list.append(temp_list)
            self.draw1.text((x, y), words[i], fill=(0, 0, 0), font=self.font)

        return self.img1, label_list


def prepare_data():
        img_char = ImageChar()
        images = []
        labels = []
        for i in range(50000):
            chinese_img_PIL, label_list = img_char.rand_img_label()
            np_img = np.asarray(chinese_img_PIL)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
            ret, np_img = cv2.threshold(np_img, 127, 255, cv2.THRESH_BINARY_INV)
            np_img = np_img / 255
            images.append(np_img.tolist())
            labels.append(label_list)
            if i % 200 == 0:
                print(i, end='\r')
        labels = np.array(labels)
        np.save('trainLab0.npy', labels)
        images = np.array(images)
        np.save('trainImg0.npy', images)

if __name__ == '__main__':
    # 训练第一个网络
    prepare_data()