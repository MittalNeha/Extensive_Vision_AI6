# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch

import util.misc as utils

from models import build_model
from datasets.materials import make_coco_transforms

import matplotlib.pyplot as plt
import time
import json
from collections import defaultdict
import textwrap

import itertools
import seaborn as sns
import io
from panopticapi.utils import id2rgb, rgb2id

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='face')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.5, type=float)

    parser.add_argument('--annot_path', type=str)

    return parser


#function to put label for each bound box
def drawBoxLabel(img, bbox, color, label):
    text_color = (0,0,0)
    # color = (0, 0, 255)
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = x1+bbox[2], y1+bbox[3]
    # For bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Prints the text.    
    img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + 20), color, -1)
    img = cv2.putText(img, label, (x1, y1 + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

def putWrappedText(img, pos, text, text_width=35, font = cv2.FONT_HERSHEY_SIMPLEX, text_color=(0,0,0)):
    wrapped_text = textwrap.wrap(text, width=text_width)
    X,Y = pos
    font_size = 0.7
    font_thickness = 2

    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 10

        y = Y + i * gap
        x = X

        cv2.putText(img, line, (x, y), font,
                    font_size, 
                    text_color, 
                    font_thickness, 
                    lineType = cv2.LINE_AA)

def save_subplots(out_file, images, title_text, subtext):

    final_img = []
    canvas_width = 800

    for img, text, subtext in zip(images, title_text, subtext):
        height, width, ch = img.shape
        # new_width, new_height = int(width + width/20), int(height + height/8)
        canvas_height = int(canvas_width*(height/width))

        # Crate a new canvas with new width and height.
        canvas = np.ones((canvas_height+60, canvas_width, ch), dtype=np.uint8) * 150

        # New replace the center of canvas with original image
        padding_top, padding_left = 60, 10
        img_width = canvas_width - 2*padding_left
        img_height = canvas_height - 2*padding_top
        if padding_top + img_height < canvas_height and padding_left + img_width < canvas_width:
            canvas[padding_top:padding_top + img_height, padding_left:padding_left + img_width] = cv2.resize(img, (img_width, img_height), cv2.INTER_AREA)
        else:
            print("The Given padding exceeds the limits.")

        img = cv2.putText(canvas.copy(), text, (int(0.25*canvas_width), 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        putWrappedText(img, (int(0.1*canvas_width), canvas_height-30), subtext, text_width=65,text_color=(0,255,255))
        # img = cv2.putText(img, subtext, (int(0.1*canvas_width), canvas_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        final_img.append(img)
        # img2 = cv2.putText(canvas.copy(), text2, (int(0.25*width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, np.array([255, 0, 0]))

    final = cv2.hconcat(final_img)
    cv2.imwrite(out_file, final)

@torch.no_grad()
def infer_compare(images_path, model, imgToAnns, device, output_path):
    model.eval()
    duration = 0
    CLASSES = ['N/A', 'misc_stuff', 'banner', 'blanket', 'bridge', 'cardboard', 'counter',
    'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
    'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road',
    'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
    'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged',
    'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged',
    'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged', 'structural_steel_-_channel', 'aluminium_frames_for_false_ceiling',
    'dump_truck___tipper_truck', 'lime', 'water_tank', 'hot_mix_plant', 'adhesives',
    'aac_blocks', 'texture_paint', 'transit_mixer', 'metal_primer', 'fine_aggregate',
    'skid_steer_loader_(bobcat)', 'rmu_units', 'enamel_paint', 'cu_piping', 'vcb_panel',
    'hollow_concrete_blocks', 'chiller', 'rcc_hume_pipes', 'wheel_loader', 'emulsion_paint',
    'grader', 'refrigerant_gas', 'smoke_detectors', 'fire_buckets', 'interlocked_switched_socket',
    'glass_wool', 'control_panel', 'river_sand', 'pipe_fittings', 'concrete_mixer_machine',
    'threaded_rod', 'vitrified_tiles', 'vrf_units', 'concrete_pump_(50%)', 'sanitary_fixtures',
    'marble', 'split_units', 'fire_extinguishers', 'hydra_crane', 'hoist', 'junction_box',
    'wood_primer', 'switch_boards_and_switches', 'distribution_transformer', 'ahus', 'rmc_batching_plant']
    num_images = 0
    for img_sample in images_path:
        num_images += 1
        if num_images > 100:
            break
        filename = os.path.basename(img_sample)
        n = ''.join(x for x in filename if x.isdigit())
        img_id = int(n)
        # print("processing...{}".format(filename))
        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        print(orig_image.size)
        if orig_image.mode == 'RGBA':
            background = Image.new("RGB", orig_image.size, (255, 255, 255))
            background.paste(orig_image, mask=orig_image.split()[3])
            orig_image = background.copy()
            background.close()
        transform = make_coco_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)


        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > args.thresh

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
        probas = probas[keep].cpu().data.numpy()

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]

        if len(bboxes_scaled) == 0:
            print('0 bboxes')
            continue

        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        p_text = None
        for p, box in zip(probas, bboxes_scaled):
        # for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            x, y = bbox[0], bbox[1]
            w, h = bbox[2], bbox[3]
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            # print(text)
            drawBoxLabel(img, bbox, (0,255,0), text)
            if p_text is None:
                p_text = "{}, {}".format(p_text, text)
            else:
                p_text = text

        # img_gt = orig_image.copy()
        img_gt = np.array(orig_image)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        gt_text = None
        for img_ann in imgToAnns[img_id]:
            bbox = np.array(img_ann['bbox'])

            bbox = bbox.astype(np.int32)
            cl = img_ann['category_id']
            text = f'{CLASSES[cl]}'
            
            drawBoxLabel(img_gt, bbox, (0,255,255), text)
            # print(text)
            if gt_text is not None: 
                gt_text = "{}, {}".format(gt_text, text)
            else:
                gt_text = text

        # out_img = cv2.hconcat([cv2.cvtColor(np.array(orig_image), cv2.COLOR_BGR2RGB),img_gt, img])
        img_save_path = os.path.join(output_path, filename)
        
        save_subplots(img_save_path, [img_gt, img],
        ["Ground Truth", "Predicted"], [gt_text, p_text])

        # save_subplots(img_save_path, [cv2.cvtColor(np.array(orig_image), cv2.COLOR_BGR2RGB),img_gt, img],
        # ["input image", "Ground Truth", "Predicted"], ["", gt_text, p_text])

        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))
        
    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


@torch.no_grad()
def infer_segm(images_path, model, postprocessor, device, output_path):
    model.eval()
    duration = 0
    CLASSES = ['N/A', 'misc_stuff', 'banner', 'blanket', 'bridge', 'cardboard', 'counter',
               'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
               'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road',
               'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
               'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
               'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged',
               'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged',
               'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged',
               'wall-other-merged', 'rug-merged', 'structural_steel_-_channel', 'aluminium_frames_for_false_ceiling',
               'dump_truck___tipper_truck', 'lime', 'water_tank', 'hot_mix_plant', 'adhesives',
               'aac_blocks', 'texture_paint', 'transit_mixer', 'metal_primer', 'fine_aggregate',
               'skid_steer_loader_(bobcat)', 'rmu_units', 'enamel_paint', 'cu_piping', 'vcb_panel',
               'hollow_concrete_blocks', 'chiller', 'rcc_hume_pipes', 'wheel_loader', 'emulsion_paint',
               'grader', 'refrigerant_gas', 'smoke_detectors', 'fire_buckets', 'interlocked_switched_socket',
               'glass_wool', 'control_panel', 'river_sand', 'pipe_fittings', 'concrete_mixer_machine',
               'threaded_rod', 'vitrified_tiles', 'vrf_units', 'concrete_pump_(50%)', 'sanitary_fixtures',
               'marble', 'split_units', 'fire_extinguishers', 'hydra_crane', 'hoist', 'junction_box',
               'wood_primer', 'switch_boards_and_switches', 'distribution_transformer', 'ahus', 'rmc_batching_plant']
    num_images = 0
    for img_sample in images_path:
        num_images += 1
        # if num_images > 100:
        #     break
        filename = os.path.basename(img_sample)
        n = ''.join(x for x in filename if x.isdigit())
        img_id = int(n)
        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        
        if orig_image.mode == 'RGBA':
            background = Image.new("RGB", orig_image.size, (255, 255, 255))
            background.paste(orig_image, mask=orig_image.split()[3])
            orig_image = background.copy()
            background.close()
        transform = make_coco_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        # compute the scores, excluding the "no-object" class (the last one)
        scores = outputs["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
        # threshold the confidence
        keep = scores > args.thresh

        # outputs["pred_logits"] = outputs["pred_logits"].cpu()
        # outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        #
        # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # # keep = probas.max(-1).values > 0.85
        # keep = probas.max(-1).values > args.thresh

        # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
        # probas = probas[keep].cpu().data.numpy()
        result = postprocessor(outputs, torch.as_tensor(image.shape[-2:]).unsqueeze(0))[0]

        palette = itertools.cycle(sns.color_palette())

        # The segmentation is stored in a special-format png
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
        # We retrieve the ids corresponding to each mask
        panoptic_seg_id = rgb2id(panoptic_seg)
        unique_ids = np.unique(panoptic_seg_id)
        # print(np.unique(panoptic_seg_id))
        if len(unique_ids) <= 1:
            infer_time = end_t - start_t
            continue

        # Finally we color each mask individually
        panoptic_seg[:, :, :] = 0
        for id in range(panoptic_seg_id.max() + 1):
            panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255
        img_save_path = os.path.join(output_path, filename)

        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (panoptic_seg.shape[-2], panoptic_seg.shape[-3]), cv2.INTER_NEAREST)
        save_subplots(img_save_path, [img, panoptic_seg],
        ["Ground Truth", "Predicted"], ["", ""])

        infer_time = end_t - start_t
        duration += infer_time
        print("Processed...{} ({:.3f}s)".format(filename, infer_time))

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(args.data_path)

    #Form imgToAnn
    f = open(args.annot_path)
    json_input = json.load(f)
    f.close()
    imgToAnns = defaultdict(list)

    for annotations in json_input['annotations']:
        imgToAnns[annotations['image_id']].append(annotations)

    if args.masks:
        #This is the segmentation model
        infer_segm(image_paths, model, postprocessors['panoptic'], device, args.output_dir)
    else:
        infer_compare(image_paths, model, imgToAnns, device, args.output_dir)
