import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.timer import Timer

parser = argparse.ArgumentParser(description="Retinaface")

parser.add_argument(
    "-m",
    "--trained_model",
    type=str,
    help="Trained state_dict file path to open",
)
parser.add_argument(
    "--network", default="resnet50", help="Backbone network mobile0.25 or resnet50"
)
parser.add_argument(
    "--save_folder",
    default="input/detection-results-all",
    type=str,
    help="Dir to save results",
)
parser.add_argument("--cpu", action="store_true", default=False, help="Use cpu inference")
parser.add_argument("--dataset", default="kouzhao", type=str, help="dataset")
parser.add_argument("--type", default="test", type=str, help="test or val")

parser.add_argument(
    "--confidence_threshold", default=0.1, type=float, help="confidence_threshold"
)
parser.add_argument("--top_k", default=5000, type=int, help="top_k")
parser.add_argument("--nms_threshold", default=0.4, type=float, help="nms_threshold")
parser.add_argument("--keep_top_k", default=750, type=int, help="keep_top_k")
parser.add_argument(
    "-s",
    "--save_image",
    action="store_true",
    default=False,
    help="show detection results",
)
parser.add_argument("--vis_thres", default=0, type=float, help="visualization_threshold")
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print("Missing keys:{}".format(len(missing_keys)))
    print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
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
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def preprocess(img, input_h=960, input_w=960):
    h, w, _ = img.shape
    # Calculate width and height and paddings
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    # Pad the short side with (128,128,128)
    img = cv2.copyMakeBorder(
        img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    img = img.astype(np.float32)
    return img


def postprocess(dets, origin_h, origin_w, input_h=960, input_w=960):
    r_w = input_w / origin_w
    r_h = input_h / origin_h
    if r_h > r_w:
        dets[:, 1] -= (input_h - r_w * origin_h) / 2
        dets[:, 3] -= (input_h - r_w * origin_h) / 2
        dets[:, :4] /= r_w
    else:
        dets[:, 0] -= (input_w - r_h * origin_w) / 2
        dets[:, 2] -= (input_w - r_h * origin_w) / 2
        dets[:, :4] /= r_h
    return dets


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print("Finished loading model!")
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # save file
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # testing dataset
    test_dataset = []
    with open(
        os.path.join("data", args.dataset, args.type, "labels.txt"), "r", encoding="utf-8"
    ) as f:
        for line in f:
            test_dataset.append(
                os.path.join(
                    "data", args.dataset, args.type, "images", line.split("\t")[0]
                )
            )

    num_images = len(test_dataset)

    # testing scale
    resize = 1

    for i, image_path in enumerate(test_dataset):
        save_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        with open(os.path.join(args.save_folder, save_name), "w", encoding="utf-8") as f:
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            origin_h, origin_w, _ = img_raw.shape
            img = preprocess(img_raw)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([im_width, im_height, im_width, im_height])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            loc, conf = net(img)  # forward pass

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            conf = conf.squeeze(0).data.cpu().numpy()
            face_scores = conf[:, 1]
            mask_scores = conf[:, 2]

            scores = np.where(face_scores > mask_scores, face_scores, mask_scores)
            face_inds = np.where(face_scores > mask_scores, 1, 0)

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]
            face_inds = face_inds[inds]

            # keep top-K before NMS
            # order = scores.argsort()[::-1][:args.top_k]
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            scores = scores[order]
            face_inds = face_inds[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                np.float32, copy=False
            )

            dets = postprocess(dets, origin_h, origin_w)

            keep = py_cpu_nms(dets, args.nms_threshold)

            dets = dets[keep, :]
            face_inds = face_inds[keep]

            print("im_detect: {:d}/{:d}".format(i + 1, num_images))
            for j, b in enumerate(dets):
                if face_inds[j] > 0:
                    f.write(
                        "face "
                        + str(b[4])
                        + " "
                        + str(int(b[0]))
                        + " "
                        + str(int(b[1]))
                        + " "
                        + str(int(b[2]))
                        + " "
                        + str(int(b[3]))
                        + "\n"
                    )
                else:
                    f.write(
                        "mask "
                        + str(b[4])
                        + " "
                        + str(int(b[0]))
                        + " "
                        + str(int(b[1]))
                        + " "
                        + str(int(b[2]))
                        + " "
                        + str(int(b[3]))
                        + "\n"
                    )
            # show image
            if args.save_image:
                for j, b in enumerate(dets):
                    if b[4] < args.vis_thres:
                        continue
                    if face_inds[j] > 0:
                        # face
                        text = "face {:.2f}".format(b[4])
                        plot_one_box(b, img_raw, (0, 0, 255), text)
                    else:
                        # mask
                        text = "mask {:.2f}".format(b[4])
                        plot_one_box(b, img_raw, (255, 0, 0), text)

                # save image
                os.makedirs("./results/", exist_ok=True)
                name = os.path.join("./results", str(i) + ".jpg")
                cv2.imwrite(name, img_raw)