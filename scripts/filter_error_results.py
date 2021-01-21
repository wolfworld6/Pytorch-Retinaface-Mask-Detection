import argparse
import glob
import json
import math
import os
import shutil
import sys
import cv2
import numpy as np
import random


def iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def plot_one_box(x, img, color=None, label=None, line_thickness=None, ground_truth=False):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        if ground_truth:
            cv2.putText(
                img,
                label,
                (c1[0], c2[1] - 2),
                0,
                tl / 3,
                [225, 0, 0],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [0, 0, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )


def save_error_image(
    image_dir, basename, save_dir, detection_results, ground_truths, is_num_error=False
):
    img_path = os.path.join(image_dir, basename)
    save_path = os.path.join(save_dir, basename)
    print(img_path)
    img = cv2.imread(img_path)
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    if is_num_error:
        cv2.putText(
            img,
            "num error",
            (0, 30),
            0,
            tl / 3,
            [0, 0, 0],
            thickness=max(tl - 1, 1),
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.putText(
            img,
            "cls error",
            (0, 30),
            0,
            tl / 3,
            [0, 0, 0],
            thickness=max(tl - 1, 1),
            lineType=cv2.LINE_AA,
        )
    for detection_result in detection_results:
        plot_one_box(
            detection_result[2:6],
            img,
            (0, 0, 255),
            "{}: {:.2f}".format(detection_result[0], detection_result[1]),
        )
    for ground_truth in ground_truths:
        plot_one_box(
            ground_truth[1:5], img, (255, 0, 0), ground_truth[0], ground_truth=True
        )
    cv2.imwrite(save_path, img)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    ground_truth_txts = glob.glob(os.path.join(args.ground_truth, "*.txt"))
    num_error = 0
    cls_error = 0
    for ground_truth_txt in ground_truth_txts:
        basename = os.path.basename(ground_truth_txt)
        image_id = os.path.splitext(basename)[0]
        detection_results = []
        with open(
            os.path.join(args.detection_result, image_id + ".txt"),
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                items = line.strip().split(" ")
                class_name = items[0]
                score = float(items[1])
                bbox_dt = list(map(int, items[2:6]))
                detection_results.append([class_name, score, *bbox_dt])
        ground_truths = []
        with open(ground_truth_txt, "r", encoding="utf-8") as f:
            for line in f:
                items = line.strip().split(" ")
                class_name = items[0]
                bbox_gt = list(map(int, items[1:]))
                ground_truths.append([class_name, *bbox_gt])
        # different count
        if len(detection_results) != len(ground_truths):
            save_error_image(
                args.image_dir,
                image_id + ".jpg",
                args.save_dir,
                detection_results,
                ground_truths,
                is_num_error=True,
            )
            num_error += 1
            continue

        for ground_truth in ground_truths:
            max_iou = -1
            idx = -1
            for i, detection_result in enumerate(detection_results):
                cur_iou = iou(ground_truth[1:5], detection_result[2:6])
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    idx = i
            if ground_truth[0] == detection_results[idx][0] and max_iou > args.overlap:
                max_iou = -1
                continue
            else:
                save_error_image(
                    args.image_dir,
                    image_id + ".jpg",
                    args.save_dir,
                    detection_results,
                    ground_truths,
                )
                cls_error += 1
                print(max_iou, ground_truth[0], detection_results[idx][0])
                break
    print("total", len(ground_truth_txts))
    print("num error", num_error)
    print("cls error", cls_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", type=str, help="original image dir")
    parser.add_argument("-d", "--detection_result", type=str, help="detection result")
    parser.add_argument("-g", "--ground_truth", type=str, help="ground truth")
    parser.add_argument("-s", "--save_dir", type=str, help="path to save result")
    parser.add_argument("-o", "--overlap", type=float, default=0.5, help="min overlap")
    args = parser.parse_args()
    main(args)