from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class LoadMnistDataSet():
    data = input_data.read_data_sets("MNIST_data", reshape=False)
    x_train, y_train = data.train.images, data.train.labels
    x_validation, y_validation = data.validation.images, data.validation.labels
    x_test,y_test = data.test.images, data.test.labels

    def __init__(self):
        assert (len(self.x_train) == len(self.y_train))
        assert (len(self.x_validation) == len(self.y_validation))
        assert (len(self.x_test) == len(self.y_test))

        print("Imported MNIST Dataset")
        print("Image Shape: {}".format(self.x_train[0].shape))
        print()
        print("Training Set:   {} samples".format(len(self.x_train)))
        print("Validation Set: {} samples".format(len(self.x_validation)))
        print("Test Set:       {} samples".format(len(self.x_test)))

    def pad(self, requested_xDim, requested_yDim):
        """
        pad dataset width and height with zeros to match dimensions
        :param requested_xDim: width after padding
        :param requested_yDim: height after padding
        :return:
        """
        width,height,channels = self.x_train[0].shape
        assert (requested_xDim > width) , "Must be greater than 28"
        assert (requested_yDim > height) , "Must be greater than 28"
        print("padding to:", (requested_xDim, requested_yDim))

        xPad = requested_xDim - width
        x0_pad = xPad//2
        x1_pad = xPad - x0_pad

        yPad = requested_yDim - height
        y0_pad = yPad // 2
        y1_pad = yPad - y0_pad

        self.x_train    = np.pad(self.x_train,(
            (0,0),
            (x0_pad,x1_pad),
            (y0_pad, y1_pad),
            (0,0)
        ),'constant')
        self.x_validation = np.pad(self.x_validation, (
            (0, 0),
            (x0_pad, x1_pad),
            (y0_pad, y1_pad),
            (0, 0)
        ),'constant')
        self.x_test = np.pad(self.x_test, (
            (0, 0),
            (x0_pad, x1_pad),
            (y0_pad, y1_pad),
            (0, 0)
        ),'constant')

        print("New Image Shape: {}".format(self.x_train[0].shape))
    def show_random(self):
        """
        Shows a random image and label
        :return:
        """
        index = random.randint(0, len(self.x_train))
        image = self.x_train[index].squeeze()

        plt.figure(figsize=(1,1))
        plt.imshow(image, cmap="gray")
        print(self.y_train[index])
        plt.show()
    def shuffle(self):
        """
        shuffles dataset
        :return:
        """
        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)





if __name__ == "__main__":
    print("\t\tDebug MNIST>>")
    mnist = LoadMnistDataSet()
    mnist.pad(33,33)
    mnist.show_random()


