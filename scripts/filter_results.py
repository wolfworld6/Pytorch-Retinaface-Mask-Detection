import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source_folder",
    type=str,
    default="input/detection-results-all",
    help="dest folder",
)
parser.add_argument(
    "-d",
    "--dest_folder",
    type=str,
    default="input/detection-results",
    help="dest folder",
)
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.5,
    help="threshold",
)
args = parser.parse_args()


os.makedirs(args.dest_folder, exist_ok=True)
txts = glob.glob(os.path.join(args.source_folder, "*.txt"))
for txt in txts:
    basename = os.path.basename(txt)
    with open(os.path.join(args.dest_folder, basename), "w", encoding="utf-8") as g:
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                items = line.strip().split(" ")
                if float(items[1]) < args.threshold:
                    continue
                g.write(line)
