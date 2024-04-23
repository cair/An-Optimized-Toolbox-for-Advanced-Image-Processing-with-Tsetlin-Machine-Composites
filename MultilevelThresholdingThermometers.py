# Multilevel algorithm from https://medium.com/analytics-vidhya/multilevel-thresholding-for-image-segmentation-d5805ad596b7

import numpy as np
import math

from tqdm import tqdm


def multi_level_tresholding(image, resolution=8):
    img = image  # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    a = 0
    b = 255
    n = resolution  # 8 # number of thresholds (better choose even value)
    k = 0.7  # free variable to take any positive value
    T = []  # list which will contain 'n' thresholds

    def multiThresh(img, a, b):
        if a > b:
            s = -1
            m = -1
            return m, s

        img = np.array(img)
        t1 = img >= a
        t2 = img <= b
        X = np.multiply(t1, t2)
        Y = np.multiply(img, X)
        s = np.sum(X)
        m = np.sum(Y) / s
        return m, s

    for i in range(int(n / 2 - 1)):
        img = np.array(img)
        t1 = img >= a
        t2 = img <= b
        X = np.multiply(t1, t2)
        Y = np.multiply(img, X)

        mu = np.sum(Y) / np.sum(X)

        Z = Y - mu
        Z = np.multiply(Z, X)
        W = np.multiply(Z, Z)
        sigma = math.sqrt(np.sum(W) / np.sum(X))

        T1 = mu - k * sigma
        T2 = mu + k * sigma

        x, y = multiThresh(img, a, T1)
        w, z = multiThresh(img, T2, b)

        T.append(x)
        T.append(w)

        a = T1 + 1
        b = T2 - 1
        k = k * (i + 1)

    T1 = mu
    T2 = mu + 1
    x, y = multiThresh(img, a, T1)
    w, z = multiThresh(img, T2, b)
    T.append(x)
    T.append(w)
    T.sort()
    return T


# This one. This one takes 1 minute to binarize the whole thing.
def mlt_temp(image_array, resolution=8):
    image_array_dynamic_temp = np.empty(
        (
            image_array.shape[0],
            image_array.shape[1],
            image_array.shape[2],
            image_array.shape[3],
            resolution,
        ),
        dtype=np.uint8,
    )

    for image in tqdm(range(image_array.shape[0])):
        for layer in range(image_array.shape[3]):
            thresholds = multi_level_tresholding(
                image_array[image, :, :, layer], resolution
            )
            # print(thresholds)
            for z, threshold in enumerate(thresholds):
                image_array_dynamic_temp[image, :, :, layer, z] = (
                    image_array[image, :, :, layer] >= threshold
                )

    image_array_dynamic_temp = image_array_dynamic_temp.reshape(
        (
            image_array.shape[0],
            image_array.shape[1],
            image_array.shape[2],
            3 * resolution,
        )
    )
    return image_array_dynamic_temp


if __name__ == "__main__":
    # import numpy as np
    from keras.datasets import cifar10

    # import cv2
    # from tmu.models.classification.vanilla_classifier import TMClassifier
    # from logger import Logger
    # import random
    # from tqdm import tqdm
    # from itertools import product
    # from itertools import combinations

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    print(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])
    print("testing mlt_temp")
    a = mlt_temp(X_train_org, 8)
    print(a)
