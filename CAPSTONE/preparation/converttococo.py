import json
from copy import deepcopy

def convert_panoptic_to_coco(in_path, out_path):
    f = open(in_path)
    json_data = json.load(f)
    f.close()

    json_data_bbox = deepcopy(json_data)

    annotations = []
    ANN_ID = 1
    for ann in json_data['annotations']:
        for seg in ann['segments_info']:
            obj_dic = deepcopy(seg)
            obj_dic['id'] = ANN_ID
            ANN_ID += 1
            obj_dic['image_id'] = ann['image_id']
            obj_dic['iscrowd'] = 0
            annotations.append(obj_dic)


    json_data_bbox['annotations'] = annotations



    out_file = open(out_path, "w")
    json.dump(json_data_bbox, out_file, indent = 4)
    out_file.close()
convert_panoptic_to_coco('coco_panoptic/annotations/train_panoptic.json', 'coco_panoptic/annotations/train_coco.json')
convert_panoptic_to_coco('coco_panoptic/annotations/val_panoptic.json', 'coco_panoptic/annotations/val_coco.json')