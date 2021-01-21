import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", type=str, help="source anotation")
parser.add_argument(
    "-d", "--dest_folder", type=str, default="input/ground-truth", help="dest folder"
)

args = parser.parse_args()
class_map = {"0": "background", "1": "face", "2": "mask"}
os.makedirs(args.dest_folder, exist_ok=True)

with open(args.source, "r", encoding="utf-8") as f:
    for line in f:
        items = line.strip().split("\t")
        img_path = items[0]
        locs = items[1]
        labels = []
        with open(
            os.path.join(args.dest_folder, os.path.splitext(img_path)[0] + ".txt"), "w"
        ) as g:
            for loc in locs.split(" "):
                l = loc.split(",")
                # class_name x1 y1 x2 y2
                g.write(
                    class_map[l[4]]
                    + " "
                    + l[0]
                    + " "
                    + l[1]
                    + " "
                    + l[2]
                    + " "
                    + l[3]
                    + "\n"
                )
