import os
import glob
import pandas as pd
import numpy as np
from keras.datasets import cifar10


(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

class_sums_directory = "class_sums/"


class Scores:
    def __init__(self, filename, ticker):
        self.filename = filename
        self.score = np.loadtxt(
            filename,
            delimiter=",",
        )
        self.acc = self._calculate_acc()
        self.ticker = ticker
        self.weight = 1
        self.max = np.max(self.score)
        self.min = np.min(self.score)

    def _calculate_acc(self):
        votes = np.zeros(self.score.shape, dtype=np.float32)
        for i in range(Y_test.shape[0]):
            votes[i] += 1.0 * self.score[i] / (np.max(self.score) - np.min(self.score))
        Y_test_predicted = votes.argmax(axis=1)
        return (Y_test_predicted == Y_test).mean()


def extract_features_from_name(name):
    split = name.split("_")

    method = None
    epoch = None
    clauses = None
    T = None
    s = None
    patch_size = None
    resolution = None
    max_included_literals = None
    weighted_clauses = None

    method = split[0] if len(split) > 0 else None
    epoch = split[1] if len(split) > 1 else None
    clauses = split[2] if len(split) > 2 else None
    T = split[3] if len(split) > 3 else None
    s = split[4] if len(split) > 4 else None
    patch_size = split[5] if len(split) > 5 else None
    if len(split) >= 8:
        if len(split) == 9:
            resolution = split[6]
            max_included_literals = split[7]
            weighted_clauses = split[8]
        elif len(split) == 8:
            max_included_literals = split[6]
            weighted_clauses = split[7]
        else:
            print(f"Unexpected filename structure: {name}")

    print(
        f"method: {method}, epoch: {epoch}, clauses: {clauses}, T: {T}, s: {s}, patch size: {patch_size}, resolution: {resolution}, max_included_literals: {max_included_literals}, weighted_clauses: {weighted_clauses}"
    )
    return (
        method,
        epoch,
        clauses,
        T,
        s,
        patch_size,
        resolution,
        max_included_literals,
        weighted_clauses,
    )


def extract_epoch(name):
    split = name.split("_")
    try:
        return int(split[1])
    except ValueError:
        print(f"Skipping file due to unexpected format: {name}")
        return None


def extract_method_and_setup(name):
    split = name.split("_")
    method = split[0]
    # epoch = int(split[1])
    setup = "_".join(split[2:])
    return method, setup


unique_setups = list(
    set(
        [
            extract_method_and_setup(os.path.basename(x))
            for x in glob.glob(class_sums_directory + "*")
        ]
    )
)


def extract_last_25_epochs(method, setup):
    epochs = [
        extract_epoch(os.path.basename(x))
        for x in glob.glob(class_sums_directory + method + "*" + setup)
    ]
    epochs.sort()
    last_25_epochs = epochs[-25:]

    if last_25_epochs != list(range(250 - 25 + 1, 250 + 1)):
        print()
        print("Something went wrong")
        print(last_25_epochs)
        print(list(range(len(epochs) - 25 + 1, len(epochs) + 1)))
        print(method, setup)
        print()

    names = []
    for epoch in last_25_epochs:
        names.append(method + f"_{epoch}_" + setup)
    return names


df = pd.DataFrame(
    columns=[
        "method",
        "setup",
        "clauses",
        "T",
        "s",
        "patch_size",
        "max_included_literals",
        "weighted_clauses",
    ]
    + [f"Accuracy_{i}" for i in range(226, 251)]
)

print(df)
for i, combination in enumerate(unique_setups):
    print(i, combination)

    method, setup = combination

    names = extract_last_25_epochs(method, setup)

    print(names)
    print(len(names))

    accs = [np.nan] * 25

    # df = pd.DataFrame(columns=["method", "setup", "clauses", "T", "s", "patch_size", "resolution", "max_included_literals", "weighted_clauses"] + list(range(226, 251)))
    for idx, name in enumerate(names):
        score = Scores(class_sums_directory + name, name)
        accs[idx] = score.acc
        print(name, score.acc)

    print(accs)
    print(len(accs))
    print(len(names))
    print(len(accs) == len(names))
    print()
    (
        method,
        epoch,
        clauses,
        T,
        s,
        patch_size,
        resolution,
        max_included_literals,
        weighted_clauses,
    ) = (
        extract_features_from_name(names[0]) if names else ""
    )
    print(
        method,
        epoch,
        clauses,
        T,
        s,
        patch_size,
        resolution,
        max_included_literals,
        weighted_clauses,
    )
    print()
    print("Creating dataframe")
    print()

    df.loc[i] = [
        method,
        setup,
        clauses,
        T,
        s,
        patch_size,
        max_included_literals,
        weighted_clauses,
    ] + accs

print(df)

df.to_csv("last_25_epochs.csv", index=False)
