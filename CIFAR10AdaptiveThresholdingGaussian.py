"""Copyright (c) 2023 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10
import cv2
from time import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=1500, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--patch_size", default=10, type=int)

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

    X_train = np.copy(X_train_org)
    X_test = np.copy(X_test_org)

    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
            X_train[i, :, :, j] = cv2.adaptiveThreshold(
                X_train_org[i, :, :, j],
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                3,
                2,
            )

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
            X_test[i, :, :, j] = cv2.adaptiveThreshold(
                X_test_org[i, :, :, j],
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                3,
                2,
            )

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
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
        stop_testing = time()

        result_test = 100 * (Y_test_scores.argmax(axis=1) == Y_test).mean()

        print(
            "#%d Adaptive Thresholding Gaussian Accuracy: %.2f%% Training: %.2fs Testing: %.2fs"
            % (
                epoch + 1,
                result_test,
                stop_training - start_training,
                stop_testing - start_testing,
            )
        )

        np.savetxt(
            "class_sums/CIFAR10AdaptiveThresholdingGaussian_%d_%d_%d_%.1f_%d_%d_%d.txt"
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