import json
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--age", type=str)

args = parser.parse_args()

for use in ["train", "test"]:
    file_list = glob.glob("./processed/" + args.age + "/" + use + "/*.json")

    label = {}

    for file in file_list:
        with open(file, "rb") as f:
            data = json.load(f)
        tmp = file.split("/")[-1].split(".")[0]

        label[tmp] = {
            "has_skeleton": True,
            "label": data["label"],
            "label_index": data["label_index"],
        }

    with open(
        "./processed/" + args.age + "/" + use + "_label.json", "wt"
    ) as f:
        json.dump(label, f, ensure_ascii=False)
