
CLASSES = ['N/A',
           'misc_stuff']
# map coco categories to custom
# Get the category list from the panoptic_val2017.json

# Format for writing categories json
LABELS = [{"supercategory": "", "isthing": 0, "id": 1, "name": "misc_stuff"}]


def getCocoCats(inputJsonFile):
    f = open(inputJsonFile)
    cocoJson = json.load(f)
    cocoCats = cocoJson["categories"]

    max_catid = cocoCats[-1]["id"]
    cocoMap = [None] * (max_catid+1)

    newCatId = 2
    for idx, cat in enumerate(cocoCats):
        if cat["isthing"] == 1:
            cocoMap[cat["id"]] = 1
        else:
            cocoMap[cat["id"]] = newCatId
            CLASSES.append(cat["name"])
            cat["id"] = newCatId
            LABELS.append(cat)
            newCatId += 1

    return cocoMap, newCatId

original_coco_file = "/home/neha/Downloads/panoptic_annotations_trainval2017/annotations/panoptic_val2017.json"
cocoMap, num_cats = getCocoCats(original_coco_file)