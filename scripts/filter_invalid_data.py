import os

txt_path = "data/kouzhao/val/labels.txt"
new_path = "data/kouzhao/val/labels1.txt"
with open(new_path, "w", encoding="utf-8") as g:
    with open(txt_path, "r", encoding="utf-8") as f:
        i = 0
        for line in f:
            keep = True
            items = line.strip().split("\t")
            img_path = items[0]
            locs = items[1]
            labels = []
            for loc in locs.split(" "):
                if len(loc.split(",")) != 5:
                    print(img_path)
                    i += 1
                    keep = False
            if keep:
                g.write(line)
        print(i)
