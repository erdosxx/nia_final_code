import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, help="the work folder for storing results"
    )
    parser.add_argument(
        "--alpha", default=1, help="weighted summation", type=float
    )

    parser.add_argument(
        "--joint-dir",
        help='Directory containing "epoch1_test_score.pkl" for joint eval results',
    )
    parser.add_argument(
        "--bone-dir",
        help='Directory containing "epoch1_test_score.pkl" for bone eval results',
    )

    arg = parser.parse_args()

    dataset = arg.dataset

    with open("./data/" + dataset + "/test_label.pkl", "rb") as label:
        label = np.array(pickle.load(label))

    with open(
        os.path.join(arg.joint_dir, "epoch1_test_score.pkl"), "rb"
    ) as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, "epoch1_test_score.pkl"), "rb") as r2:
        r2 = list(pickle.load(r2).items())

    right_num = total_num = right_num_3 = 0
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        r = r11 + r22 * arg.alpha
        rank_3 = r.argsort()[-3:]
        right_num_3 += int(int(l) in rank_3)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc3 = right_num_3 / total_num

    print("Top1 Acc: {:.4f}%".format(acc * 100))
    print("Top3 Acc: {:.4f}%".format(acc3 * 100))
