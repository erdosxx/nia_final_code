import json
import glob
import os
import numpy as np
import logging
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--age", type=str)
parser.add_argument("--label", choices={"action", "k-dst", "normal"})

args = parser.parse_args()

id_list = os.listdir("GMS/")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler("empty_file.log")
logger.addHandler(handler)


for id in id_list:
    if id[0] != args.age:
        continue

    for ac_num in range(1, 5):
        for cam_num in range(1, 5):
            frame_list = glob.glob(
                "GMS/"
                + id
                + "/"
                + id
                + "_GMS_"
                + str(ac_num)
                + "_"
                + str(cam_num)
                + "*.json"
            )

            frame_list.sort()
            video_info = {}
            skel = []
            for i, frame in enumerate(frame_list):
                try:
                    with open(frame, "rb") as f:
                        data = json.load(f)
                except:
                    logging.info(frame)
                    continue

                if i == 0:
                    if args.label == "action":
                        if int(data["metaData"]["beh_eval_code"]) < 2:
                            break
                        video_info["ID"] = str(data["metaData"]["ID"])
                        video_info["label"] = data["metaData"]["action_num"]
                        index_tmp = data["metaData"]["action_num_code"].split(
                            "-"
                        )
                        video_info["label_index"] = ac_num - 1
                    elif args.label == "k-dst":
                        video_info["ID"] = str(data["metaData"]["ID"])
                        video_info["label"] = data["metaData"]["beh_eval"]
                        video_info["label_index"] = int(
                            data["metaData"]["beh_eval_code"]
                        )
                    else:
                        video_info["ID"] = str(data["metaData"]["ID"])
                        if int(data["metaData"]["devl_eval_code"]) > 1:
                            video_info["label"] = "정상"
                            video_info["label_index"] = 0
                        if int(data["metaData"]["devl_eval_code"]) < 2:
                            video_info["label"] = "비정상"
                            video_info["label_index"] = 1
                body_points = [
                    [j for j in k.values()]
                    for k in data["labelingInfo"][0]["pose"][
                        "location"
                    ].values()
                ]
                body_points = list(np.reshape(body_points, (-1)))
                a = body_points[0::3]
                b = body_points[1::3]

                body_list = []

                body_list.append(a[10])
                body_list.append(b[10])

                body_list.append(a[3])
                body_list.append(b[3])

                body_list.append(a[14])
                body_list.append(b[14])

                body_list.append(a[12])
                body_list.append(b[12])

                body_list.append(a[4])
                body_list.append(b[4])

                body_list.append(a[7])
                body_list.append(b[7])

                body_list.append(a[8])
                body_list.append(b[8])

                body_list.append(a[1])
                body_list.append(b[1])

                body_list.append(a[5])
                body_list.append(b[5])

                body_list.append(a[0])
                body_list.append(b[0])

                body_list.append(a[11])
                body_list.append(b[11])

                body_list.append(a[17])
                body_list.append(b[17])

                body_list.append(a[13])
                body_list.append(b[13])

                body_list.append(a[6])
                body_list.append(b[6])

                body_list.append(a[2])
                body_list.append(b[2])

                body_list.append(a[15])
                body_list.append(b[15])

                body_list.append(a[16])
                body_list.append(b[16])

                body_list.append(a[9])
                body_list.append(b[9])

                skel.append({"frame_index": i + 1, "skeleton": body_list})
            if len(skel) == 0:
                continue
            video_info["data"] = skel
            with open(
                "./processed/"
                + args.age
                + "/"
                + video_info["ID"]
                + "GMS_"
                + str(ac_num)
                + "_"
                + str(cam_num)
                + ".json",
                "wt",
            ) as f:
                json.dump(video_info, f, ensure_ascii=False)
