import argparse
import os
import struct

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="resnet or mobile")
parser.add_argument("-w", "--weight", type=str, help="trained weight")
args = parser.parse_args()


def main():
    state_dict = torch.load(args.weight)

    f = open(args.model+"_retinaface.wts", "w")
    f.write("{}\n".format(len(state_dict.keys())))
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        print("key: ", k)
        print("value: ", v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


if __name__ == "__main__":
    main()
