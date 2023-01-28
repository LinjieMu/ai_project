import numpy as np
import cv2
import os


class KNN:
    """
    定义一个KNN类，所有方法在其中实现
    """

    def __init__(self, k, width=256, height=256, kind=4):
        """
        构造方法
        :param k: 考虑最近k个样本
        """
        self.k = k
        self.width = width
        self.height = height
        self.kind = kind

    def reform(self, img, is_show=False):
        """
        形状转变方法
        1. 以灰度图形式读入图片
        2. 将图片转为特定形状
        :param is_show: 是否显示图片
        :param img: 输入图像的地址
        :return: nparray形式的图片
        """
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))

        # 显示图片
        if is_show:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img

    def predict(self, img, distance='M'):
        """
        分类方法
        1. 读入并预处理待分类图片
        2. 声明记录距离和标签的向量
        3. 循环依次检验每一个样本的差异并做记录
        [0 -> bing, 1 -> liu, 2 -> zhe, 3 -> zhi]
        :param distance: 采用何种距离计算差异性，默认欧式距离
        :param img: 待分类的图片地址
        :return: 返回预测的结果标签和概率
        """
        assert distance == 'E' or distance == 'M'
        img = self.reform(img)
        dis_arr = np.array([np.inf] * self.k)  # 记录前k最小距离
        label_arr = np.zeros(self.k, dtype=int)  # 前k最小距离对应的类别
        labels = {}
        for num, di in enumerate(os.listdir("./trainset")):
            labels[num] = di
            for n, img_name in enumerate(os.listdir(f"./trainset/{di}")):
                # 读入并转换图片类型
                img1 = self.reform(f"./trainset/{di}/{img_name}", is_show=False)
                # 计算两张图片的差异性
                if distance == 'E':
                    dis = np.sum((img - img1) ** 2) / (self.height * self.width)
                elif distance == 'M':
                    dis = np.sum(np.abs(img - img1)) / (self.height * self.width)
                # 记录差异性
                ma = np.max(dis_arr)
                if dis < ma:
                    index = np.argmax(dis_arr)
                    dis_arr[index] = dis
                    label_arr[index] = num
                print(f'\r正在测试{di}类，测试进度为{n+1}/{len(os.listdir(f"./trainset/{di}"))}...', end="")
            print()
        # 结果预测
        result_arr = np.zeros(self.kind)  # 存放每个类别的可能性
        if distance == 'E':
            dis_arr = 1 / (dis_arr ** 2)
        elif distance == 'M':
            dis_arr = 1 / dis_arr
        su = np.sum(dis_arr)
        for i in range(self.k):
            result_arr[label_arr[i]] += (dis_arr[i] / su)
        return np.max(result_arr), labels[np.argmax(result_arr)]


if __name__ == "__main__":
    knn = KNN(k=5)
    name = "./zhe1.webp"    # 待分类图片地址
    p, l = knn.predict(name)
    print(f"预测标签为{l}，概率为{p :.2}")
