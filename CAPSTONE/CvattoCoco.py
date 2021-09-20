import os
import json
from collections import defaultdict
import shutil
import os
import json
from sklearn.model_selection import train_test_split
from PIL import Image

class CVATtoCOCO:
    train_image_cnt = 1
    train_annot_cnt = 1
    json_train_imgs = []
    json_train_annot = []

    val_image_cnt = 1
    val_annot_cnt = 1
    json_val_imgs = []
    json_val_annot = []

    all_categories = []
    CLASSES = []

    outdir = '/parsed_data/'
    traindir = 'materials_train/'
    testdir = 'materials_test/'
    valdir = 'materials_val/'

    def __init__(self, data_dir, split_test_size=0.10, start_cat_id = 0):

        self.rootdir = data_dir #data_dir.rsplit('/', 1)[0]
        if data_dir[-1] == '/':
            self.outdir = data_dir.rsplit('/', 2)[0] + self.outdir
        else:
            self.outdir = data_dir.rsplit('/', 1)[0] + self.outdir
            self.outdir = self.outdir + '/'
        self.test_size = split_test_size

        # os.makedirs(self.outdir,exist_ok=True)

        os.makedirs(self.outdir + self.traindir,exist_ok=True)
        os.makedirs(self.outdir + self.testdir,exist_ok=True)
        os.makedirs(self.outdir + self.valdir,exist_ok=True)

        for idx, d in enumerate(os.listdir(data_dir)):
            if os.path.isfile(data_dir + d):
                continue
            self.CLASSES.append(d)

            # read the json file
            # print(self.rootdir + d + '/coco.json')
            f = open(self.rootdir + '/' +  d + '/coco.json')
            labels = json.load(f)
            cats= self.parse_coco_dataset(labels, d, start_cat_id)
            f.close()

            # extract categories and change the category id.
            # cats_json = json.load(cats)
            cats[0]['id'] = start_cat_id
            self.all_categories.append(cats[0])
            start_cat_id +=1

    def create_file_str(self, num, n_zeros=10):
        num_str = str(int(num))
        num_str = num_str.zfill(n_zeros)
        return num_str

    def parse_coco_dataset(self, json_input, d, cat_id):
        imgToAnns = defaultdict(list)

        for annotations in json_input['annotations']:
            imgToAnns[annotations['image_id']].append(annotations)

        # split images to train and val
        X_train, X_val = train_test_split(list(imgToAnns.keys()), test_size=self.test_size, random_state=42)
        # print('val_num {}'.format(len(X_val)))
        # print(X_val)
        for img in json_input['images']:
            file_path = self.rootdir + d + '/images/' + img['file_name']
            if not os.path.exists(file_path):
                continue
            img_id = img['id']

            if (len(imgToAnns[img_id]) == 0):
                # No annotations for this image move it to test data
                print(img['file_name'])
                print("renaming {} to {}".format(self.rootdir + d + '/images/' + img['file_name'],
                                                 self.outdir + self.testdir + d + '_' + img['file_name']))
                shutil.move(self.rootdir + d + '/images/' + img['file_name'], self.outdir + self.testdir + d + '_' + img['file_name'])
                continue

            if img_id in X_train:
                # This image to be kept for train
                # update the image id in both 'images' and 'annotations'
                img['id'] = self.train_image_cnt  # some update to the image_id
                for ann in imgToAnns[img_id]:
                    ann['image_id'] = img['id']
                    ann['id'] = self.train_annot_cnt
                    ann['category_id'] = cat_id
                    self.train_annot_cnt += 1
                    self.json_train_annot.append(ann)
                #file name should be same as the id
                img['file_name'] = self.create_file_str(img['id']) + '.jpg'
                self.json_train_imgs.append(img)
                # check for image format for exceptions
                self.handle_image_exceptions(file_path, self.outdir + self.traindir + img['file_name'])

                self.train_image_cnt += 1
            else:
                # Image for val

                # update the image id in both 'images' and 'annotations'
                img['id'] = self.val_image_cnt  # some update to the image_id
                for ann in imgToAnns[img_id]:
                    ann['image_id'] = img['id']
                    ann['id'] = self.val_annot_cnt
                    ann['category_id'] = cat_id
                    self.val_annot_cnt += 1
                    self.json_val_annot.append(ann)
                img['file_name'] = self.create_file_str(img['id']) + '.jpg'
                self.json_val_imgs.append(img)
                # check for image format for exceptions
                self.handle_image_exceptions(file_path, self.outdir + self.valdir + img['file_name'])

                self.val_image_cnt += 1
        print('val_annot {}'.format(len(self.json_val_annot)))
        return json_input['categories']

    def handle_image_exceptions(self, file_path, out_file_path):
        orig_image = Image.open(file_path)
        # change format to RGB
        #also change the file type to jpg
        if orig_image.mode != 'RGB':
            orig_image = orig_image.convert('RGB')

        orig_image.save(out_file_path)
        os.remove(file_path)

    def combine(self, coco_classes, coco_categories):
        print("CVAT num classes {}".format(len(self.CLASSES)))
        coco_classes.extend(self.CLASSES)
        self.CLASSES = coco_classes
        print("combined num classes {}".format(len(self.CLASSES)))

        # coco sample categories json
        label_temp = {"name": "material",
                      "isthing": 1,
                      "supercategory": "material"}
        for cat in self.all_categories:
            lab = label_temp.copy()
            lab["id"] = cat["id"]
            lab["name"] = cat["name"]
            coco_categories.append(lab)
        self.all_categories = coco_categories

        return coco_categories


    def save_annotations(self, annotdir):
        os.makedirs(self.outdir + annotdir,exist_ok=True)

        val_res_file = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
            "categories": self.all_categories,
            "images": self.json_val_imgs,
            "annotations": self.json_val_annot
        }

        train_res_file = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
            "categories": self.all_categories,
            "images": self.json_train_imgs,
            "annotations": self.json_train_annot
        }

        json_file = self.outdir + 'annotations/val_coco.json'
        with open(json_file, "w") as f:
            json_str = json.dumps(val_res_file)
            f.write(json_str)

        json_file = self.outdir + 'annotations/train_coco.json'
        with open(json_file, "w") as f:
            json_str = json.dumps(train_res_file)
            f.write(json_str)



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
dataset = CVATtoCOCO('/home/neha/Work/eva6/capstone/data_subset/', start_cat_id=num_cats)
combine_categories = dataset.combine(CLASSES, LABELS)
print("DONE")
dataset.save_annotations('annotations/')
print(dataset.CLASSES)

# Save the mapping for the coco categories, since the indexes have changed
file = open('map_coco_categories.p', 'wb')
pickle.dump(cocoMap, file)
file.close()
