import json
import random
import pickle
import shutil
from copy import deepcopy
from collections import defaultdict, deque
from pathlib import Path

#combine the materials panoptic annotations with coco panoptic annotations.
coco_path = "annotations/panoptic_val2017.json"
coco_mask_folder = "panoptic_val2017"
materials_path_train = "coco_panoptic/annotations/train_panoptic.json"
materials_path_val = "coco_panoptic/annotations/val_panoptic.json"
materials_path = [materials_path_train, materials_path_val]
materials_mask_folder = ["panoptic_materials_train", "panoptic_materials_val"]
len_data = [400, 100]

src_mask_path = Path(coco_path.rsplit('/', 1)[0]) / coco_mask_folder
dst_mask_path = [Path(materials_path[0].rsplit('/', 1)[0]) / materials_mask_folder[0],
                 Path(materials_path[1].rsplit('/', 1)[0]) / materials_mask_folder[1]]

src_image_path = Path("val2017")
dst_image_path = [Path("coco_panoptic/materials_train/"),
                  Path("coco_panoptic/materials_val/")]

f = open(coco_path)
json_data = json.load(f)
f.close()

coco_json = deepcopy(json_data)

images = coco_json["images"]
coco_json["annotations"]

imgToAnns = defaultdict(list)

for annotations in coco_json['annotations']:
    imgToAnns[annotations['image_id']] = annotations

file = open('map_coco_categories.p', 'rb')
cocoMap = pickle.load(file)
file.close()

select_images = [[],[]]
select_anns = [[],[]]
tot_images = len(coco_json["images"])

for image_set, len in enumerate(len_data):
    rand_num = random.randint(0, tot_images)
    for i in range(len):
        img = images[rand_num]
        ann = imgToAnns[img["id"]]
        assert (ann["image_id"] == img["id"])
        shutil.copy(src_mask_path / img['file_name'].replace('.jpg', '.png'),
                    dst_mask_path[image_set] / img['file_name'].replace('.jpg', '.png'))

        shutil.copy(src_image_path / img['file_name'],
                    dst_image_path[image_set] / img['file_name'])

        #update the category_id
        for seg in ann["segments_info"]:
            cat = seg["category_id"]
            seg["category_id"] = cocoMap[cat]
        select_images[image_set].append(img)
        select_anns[image_set].append(ann)

out_path = ["combined_train.json", "combined_val.json"]
for image_set, dir in enumerate(materials_path):
    f = open(dir)
    json_data = json.load(f)
    f.close()

    panoptic_json = deepcopy(json_data)
    panoptic_json["images"].extend(select_images[image_set])
    panoptic_json["annotations"].extend(select_anns[image_set])

    dir = out_path[image_set]
    out_file = open(dir, "w")
    json.dump(panoptic_json, out_file, indent=4)
    out_file.close()