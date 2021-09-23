# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
import pickle
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

import util.box_ops as box_ops
from util.box_ops import masks_to_boxes
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

import cv2 as cv
from panopticapi.utils import id2rgb, rgb2id
try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class PostProcessPanopticInstance(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API
    Along with this is takes input as the instance segmentation mask from the custom dataset that needs to be
    overlaid on top of the mask from the output of the mask
    """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    @torch.no_grad()
    def forward(self, outputs, processed_sizes, input_segments, input_segment_labels, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            input_segments: This is a list of tuples (or torch tensors)
            input_segment_labels: This is a list of tuples of the labels of the segments that were passed in as input_segments

            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        # Load the coco to custom class mapping
        # Save the mapping for the coco categories, since the indexes have changed
        file = open('../map_coco_categories.p', 'rb')
        cocomap = pickle.load(file)
        file.close()

        def prepare_instance_mask(input_segments, raw_masks):
            # giving more weightage to this mask so that it gets priority over the other class masks

            input_segments = cv.normalize(input_segments, None, alpha=0, beta=2, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            custom_mask = cv.resize(input_segments, (raw_masks.shape[-1], raw_masks.shape[-2]),
                                    interpolation=cv.INTER_NEAREST)

            # convert to tensor
            custom_mask = torch.from_numpy(custom_mask)

            return custom_mask

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size, cur_instance_mask, cur_instance_label in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes, input_segments, input_segment_labels
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            # print("Classes before: {}".format(cur_classes))
            # Map the coco class index to the custom class index
            cur_classes = torch.IntTensor([cocomap[cl] for cl in cur_classes])
            # print("Classes after: {}".format(cur_classes))

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # print(len(cur_instance_mask))
            custom_mask = prepare_instance_mask(cur_instance_mask, cur_masks)

            # cur_classes.append(cur_instance_label)
            cur_classes = torch.cat((cur_classes, cur_instance_label))
            cur_scores = torch.cat((cur_scores, torch.Tensor([1]).to(cur_scores.device)))
            # print("Classes: {}".format(cur_classes))

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            # print("Before flatten {}".format(cur_masks.shape))
            cur_masks = cur_masks.flatten(1)
            custom_mask = torch.unsqueeze(custom_mask.flatten(), 1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area_bbox(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                #Combine these masks with the input_segments
                m_id = torch.cat((m_id, custom_mask.to(m_id.device)), 1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)
                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))
                # print(m_id.unique())

                segment_ids = m_id.unique()

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                # print("Area: {}".format(area))
                return area, seg_img, segment_ids

            # cur_masks = cur_masks.transpose(0, 1).softmax(-1)
            # print(cur_masks.shape)
            area, seg_img, segment_ids = get_ids_area_bbox(cur_masks, cur_scores, dedup=True)
            # print("Scores {}".format(cur_scores))

            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    # print(cur_masks.shape)
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small[:-1].any().item():
                        cur_scores = cur_scores[:-1]
                        #ignore the custom class added at the last index
                        cur_scores = cur_scores[~filtered_small[:-1]]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small[:-1]]
                        #Add the score for the last class
                        cur_scores = torch.cat((cur_scores, torch.Tensor([1]).to(cur_scores.device)))
                        area, seg_img, segment_ids = get_ids_area_bbox(cur_masks, cur_scores)
                    else:
                        break
                # print(area)

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            ## -- Create segmentation ids.
            m_id = cur_masks.transpose(0, 1).softmax(-1)

            # Combine these masks with the input_segments
            m_id = torch.cat((m_id, custom_mask.to(m_id.device)), 1)

            if m_id.shape[-1] == 0:
                # We didn't detect any mask :(
                m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
            else:
                m_id = m_id.argmax(-1).view(h, w)
            # print(m_id.unique())
            #
            bboxes = []

            new_id = cur_classes*1000 + torch.Tensor(range(len(cur_classes)))
            new_id = new_id.type(torch.int64)
            # print("New id {}".format(new_id))
            m_id_copy = m_id.detach().clone()
            for seg_id in range(len(new_id)):
                seg_mask = m_id_copy == seg_id
                seg_mask = seg_mask.to(torch.device("cpu"))
                bboxes.append(masks_to_boxes(seg_mask).squeeze().int().tolist())
                # print("box {}, ({},{}) = {}".format(seg_id,h, w, masks_to_boxes(seg_mask)))
                m_id[m_id_copy == seg_id] = new_id[seg_id].to(m_id.device)

            final_h, final_w = to_tuple(target_size)
            seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
            seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)
            # ?TODO Neha: upscale the bbox to (final_w, final_h)

            np_seg_img = (
                torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
            )
            m_id = torch.from_numpy(rgb2id(np_seg_img))
            # print(m_id.unique())

            segment_ids = m_id.unique()

            area = []
            # scores = cur_scores.append
            scores = torch.cat((cur_scores, torch.Tensor([1]).to(cur_scores.device)))
            for idx, seg_id in enumerate(new_id):
                area.append(m_id.eq(seg_id).sum().item())
            # print(area)

            #Check for the area of the custom class and remove the index if it is small
            if area[-1] <4:
                cur_classes = cur_classes[:-1]
                if len(cur_classes) < len(segment_ids):
                    segment_ids = segment_ids[:-1]

                bboxes = bboxes[:-1]
                area = area[:-1]

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                # color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                segments_info.append({"id": int(segment_ids[i]),
                                      "category_id": cat,
                                      "area": a,
                                      "iscrowd": int(0),
                                      "bbox": bboxes[i]})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds


def build(args):

    is_thing_map = {i: i <= 90 for i in range(201)}
    postprocessors = {'merge': PostProcessPanopticInstance(is_thing_map, threshold=0.85)}
    return postprocessors
