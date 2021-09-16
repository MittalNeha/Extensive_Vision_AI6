# Training Custom DeTr
The combined dataset for construction materials is provided [here](https://drive.google.com/file/d/1IsK268zLnXB2Qq0X2LgNDwBZRuVwvjRx/view?usp=sharing). This dataset in in the coco format. Following steps were followed to train the model.

1. Polish the dataset:
The dataset downloaded through CVAT is such that each class has an accotations json. Hence we need to combine this such that there is one annotations file for all the training images and another for validation images. This is acieved throught [this](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/CAPSTONE/CvattoCoco.py) script. Also the images are split between test and train.
3. Train the custom dataset for bounding boxes:
Next, with the help of some changes to the detr code by facebook research https://github.com/facebookresearch/detr, a model is trained that predicts the bound boxes. 
[weights](https://drive.google.com/drive/folders/1PbBRuRYNeGTwIz6nfaeSCHEDPOIDK4aQ?usp=sharing) for this model trained for about 1o epochs. 

5. Add the coco stuffs class to the dataset and create new annotations for panoptic segmentation
This is still work in progress. 

7. Train the bounding box model for constructiion_materials + COCO stuffs class.
