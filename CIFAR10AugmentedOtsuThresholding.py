import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10
import cv2
from time import time


def horizontal_flip(image):
    return cv2.flip(image, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=3000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--patch_size", default=10, type=int)

    augmented_images = []
    augmented_labels = []

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

    for i in range(len(X_train_org)):
        image = X_train_org[i]
        label = Y_train[i]

        augmented_images.append(image)
        augmented_labels.append(label)

        augmented_images.append(horizontal_flip(image))
        augmented_labels.append(label)

    X_train_aug = np.array(augmented_images)
    Y_train = np.array(augmented_labels).reshape(-1, 1)

    X_train = np.copy(X_train_aug)
    X_test = np.copy(X_test_org)

    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])

    # Apply Otsu's threshold using Gaussian filtering
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
            ret, X_train[i, :, :, j] = cv2.threshold(
                X_train_aug[i, :, :, j], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
            ret, X_test[i, :, :, j] = cv2.threshold(
                X_test_org[i, :, :, j], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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
            "#%d Augmented Otsu Accuracy: %.2f%% Training: %.2fs Testing: %.2fs"
            % (
                epoch + 1,
                result_test,
                stop_training - start_training,
                stop_testing - start_testing,
            )
        )

        np.savetxt(
            "class_sums/CIFAR10AugmentedOtsuThresholding_%d_%d_%d_%.1f_%d_%d_%d.txt"
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
