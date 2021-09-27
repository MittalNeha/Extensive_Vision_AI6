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

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    model.eval()
    duration = 0
    CLASSES = ['misc_stuff', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 
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
    for img_sample in images_path:
        filename = os.path.basename(img_sample)
        print("processing...{}".format(filename))
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
        for p, box in zip(probas, bboxes_scaled):
        # for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            x, y = bbox[0], bbox[1]
            w, h = bbox[2], bbox[3]
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                ])
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            img = cv2.putText(
                img=img,
                text=text,
                org=(x + 5, y + int(h/2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3
            )

        out_img = cv2.hconcat([cv2.cvtColor(np.array(orig_image), cv2.COLOR_BGR2RGB), img])
        img_save_path = os.path.join(output_path, filename)
        cv2.imwrite(img_save_path, out_img)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


@torch.no_grad()
def detect(img_sample, model, device):
    model.eval()
    duration = 0
    # for img_sample in images_path:
    filename = os.path.basename(img_sample)
    print("processing...{}".format(filename))
    orig_image = Image.open(img_sample)
    w, h = orig_image.size
    # if orig_image.mode == 'RGBA':
    #     background = Image.new("RGB", orig_image.size, (255, 255, 255))
    #     background.paste(orig_image, mask=orig_image.split()[3])
    #     orig_image = background.copy()
    #     background.close()
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

    infer_time = end_t - start_t
    duration += infer_time
    print("Processing...{} ({:.3f}s)".format(filename, infer_time))

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))

    return probas, bboxes_scaled


def plot_results(img_path, prob, boxes):
    CLASSES = ['aac_blocks', 'ahus', 'lime']

    img = Image.open(img_path)

    for p, box in zip(prob, boxes):
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            ])
        bbox = bbox.reshape((4, 2))
        cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        img = cv2.putText(
            img=img,
            text=text,
            org=(bbox[0], bbox[1]),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=3.0,
            color=(125, 246, 55),
            thickness=3
        )

    filename = os.path.basename(img_path)
    img_save_path = os.path.join(args.output_dir, filename)
    cv2.imwrite(img_save_path, img)
    # cv2.imshow("img", img)
    # cv2.waitKey()


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

    infer(image_paths, model, postprocessors, device, args.output_dir)
