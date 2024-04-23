import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10
from time import time
from MultilevelThresholdingThermometers import mlt_temp
import cv2
import random


def horizontal_flip(image):
    return cv2.flip(image, 1)


def shuffle_dataset(image_array, label_array):
    pairs = list(zip(image_array, label_array))
    random.shuffle(pairs)
    image_array_rand = []
    label_array_rand = []
    for i in range(len(pairs)):
        image_array_rand.append(pairs[i][0])
        label_array_rand.append(pairs[i][1])
    return (np.asarray(image_array_rand), label_array_rand)


def batch_train(tm, batchsize, x_train, y_train):
    (x_train, y_train) = shuffle_dataset(x_train, y_train)
    x_train = np.asarray(x_train)
    for i in range(0, len(x_train), batchsize):
        X_batch = x_train[i : i + batchsize]
        Y_batch = y_train[i : i + batchsize]
        tm.fit(X_batch, Y_batch)


batchsize = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=3000, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--patch_size", default=4, type=int)

    augmented_images = []
    augmented_labels = []

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

    for i in range(len(X_train_org)):
        image = X_train_org[i]
        label = Y_train[i]

        # Original image and label
        augmented_images.append(image)
        augmented_labels.append(label)

        augmented_images.append(horizontal_flip(image))
        augmented_labels.append(label)

    X_train_aug = np.array(augmented_images)
    Y_train = np.array(augmented_labels).reshape(-1, 1)

    X_train = mlt_temp(X_train_aug)
    X_test = mlt_temp(X_test_org)

    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        patch_dim=(args.patch_size, args.patch_size),
    )

    for epoch in range(args.epochs):
        start_training = time()

        batch_train(tm, batchsize, X_train, Y_train)

        stop_training = time()

        start_testing = time()
        Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
        stop_testing = time()

        result_test = 100 * (Y_test_scores.argmax(axis=1) == Y_test).mean()

        print(
            "#%d Augmented Adaptive Color Accuracy: %.2f%% Training: %.2fs Testing: %.2fs"
            % (
                epoch + 1,
                result_test,
                stop_training - start_training,
                stop_testing - start_testing,
            )
        )

        np.savetxt(
            "class_sums/CIFAR10AugmentedAdaptiveColorThermometers_%d_%d_%d_%.1f_%d_%d_%d.txt"
            % (
                epoch + 1,
                args.num_clauses,
                args.T,
                args.s,
                args.patch_size,
                args.max_included_literals,
                args.weighted_clauses,
            ),
            Y_test_scores,
            delimiter=",",
        )
