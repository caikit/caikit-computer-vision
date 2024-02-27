# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Detectron2 postprocess tasks and functions related.
"""
# Standard
from tempfile import NamedTemporaryFile
from typing import Type, Union
import json

# Third Party
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2
import numpy as np
import pycocotools.mask as maskUtils
import shapely.geometry as spg
import torch

# First Party
from caikit.core.exceptions import error_handler
import alog

log = alog.use_channel("IMAGE_SEGMENTATION")
error = error_handler.get(log)


def maskrcnn_postprocess(
    results: tuple,
    uuid: str,
    image_id: int = 1,
    threshold=0.2,
) -> dict:
    """Postprocess task for MaskRCNN type models.

    Args:
        results (dict): COCO json from an inference task
        uuid (str): Unique identifier of image
        image_id (int, optional): Image ID to store in the task. Defaults to 1.
        threshold (float):  Threshold in range (0,1] to be used for filtering bounding box predictions.
                Default 0.2
    Returns:
        dict: Formatted COCO JSON in OCL format
    """
    if len(results) == 0:
        return {}
    image_size = results["IMAGES_SIZE"]
    instances = _convert_results_dict_to_instances(results)
    log.info(f"Processing Instances of len {len(instances)}")
    try:
        result_instances = detector_postprocess(instances, image_size[0], image_size[1])
    except IndexError as e:
        error("<CCV21811501E>", f"Length of MASKS {len(results['MASKS'])}")
        error(
            "<CCV96470531E>",
            "No instances found, setting empty annotations",
            exc_info=e,
        )
        result_instances = []
    coco_instances = instances_to_coco_json(result_instances, image_id)
    coco_json = _assemble_images_coco(
        file_name=f"{uuid}.jpg",
        height=image_size[0],
        width=image_size[1],
        image_id=image_id,
    )
    inf_coco = detectron2_to_coco(
        detectron2_out=coco_instances,
        inf_coco=coco_json,
        score_threshold=threshold,
        save_coco="",
    )
    coco_json = inf_coco.dataset
    return coco_json


def _convert_results_dict_to_instances(results: dict) -> "Instances":
    """Helper function to convert output of model to Instances.

    Args:
        results (dict): Raw COCO Dict from Detectron2.

    Returns:
        Instances: Instances object.
    """
    boxes: Boxes = Boxes(results["BOXES"])
    classes: torch.Tensor = torch.as_tensor(results["CLASSES"])
    masks: torch.Tensor = torch.as_tensor(results["MASKS"])
    scores: torch.Tensor = torch.as_tensor(results["SCORES"])
    image_size = results["IMAGES_SIZE"]
    instances = Instances(
        image_size=image_size,
        pred_boxes=boxes,
        scores=scores,
        pred_classes=classes,
        pred_masks=masks,
    )
    return instances


def _assemble_images_coco(
    file_name: str,
    height: int,
    width: int,
    image_id: int = 1,
) -> "COCO":
    """Assembles coco dictionary into file and returns COCO object.

    Args:
        file_name (str): File name
        height (int): Height of image
        width (int): Width of image
        image_id (int, optional): Optional image id. Defaults to 1.

    Returns:
        COCO: COCO Object
    """
    coco_json = dict(
        images=[
            {
                "id": image_id,
                "file_name": file_name,
                "height": height,
                "width": width,
            }
        ],
    )
    with NamedTemporaryFile(suffix=".json") as tmp_file:
        with open(tmp_file.name, "w") as f_temp:
            json.dump(coco_json, f_temp)
        coco = COCO(tmp_file.name)
    return coco


def detectron2_to_coco(
    detectron2_out: Union[dict, str],
    inf_coco: Union[Type[COCO], str],
    score_threshold: float = 0.2,
    save_coco: str = "",
) -> Type[COCO]:
    """convert detectron2 inference output to coco.

    Args:
      detectron2_out  : path to inference output .json file or dictionary with detectron2 inference output
      inf_coco        : inference coco file, required to contain 'images' and 'categories' keys
      score_threshold : minimum score threshold to keep an annotation
      save_coco       : if a save_coco is provided, then the coco annotations are saved to JSON
                        with the file name 'coco_results_poly.json' under the path provided in save_coco

    Returns:
      coco_dict : coco formatted dict of annotations
    """

    if isinstance(detectron2_out, str):
        with open(detectron2_out, "r") as f_out:
            detectron2_out = json.load(f_out)

    if isinstance(inf_coco, str):
        inf_coco = COCO(inf_coco)
    else:
        error.type_check("<CCV49366892E>", COCO, inf_coco=inf_coco)

    error.value_check(
        "<CCV62117290E>",
        ("categories" and "images" in inf_coco.dataset),
        "inf_coco should contain both 'categories' and 'images' keys",
    )

    # give ids to annotations
    # correct the placement of the score key as in coco
    ann_id = 1
    for ann in detectron2_out:
        ann["id"] = ann_id
        ann_id += 1
        ann["iscrowd"] = 0
        score = ann.pop("score")
        ann.update({"attributes": {"score": score}})

    # drop annotations with score < score_threshold
    inf_coco.dataset["annotations"] = [
        ann for ann in detectron2_out if ann["attributes"]["score"] >= score_threshold
    ]

    # convert segmentation to polygons
    inf_coco = convert_seg_poly(inf_coco)

    # delete empty segmentations and invalid segmentations
    inf_coco.dataset["annotations"] = [
        ann
        for ann in inf_coco.dataset["annotations"]
        if ann["segmentation"]
        if len(ann["segmentation"][0]) > 6
    ]

    log.info("Calculating bounding boxes and areas...")

    for idx, ann in enumerate(inf_coco.dataset["annotations"]):
        # we usually have multiple polygon parts produced from one prediction mask
        multipolygon = spg.MultiPolygon(
            [
                spg.Polygon(list(zip(poly[0::2], poly[1::2])))
                for poly in ann["segmentation"]
            ]
        )
        ann["area"] = int(multipolygon.area)

    # extract seg array from list of segs and cast to list
    for idx, ann in enumerate(inf_coco.dataset["annotations"]):
        seg = [i.tolist() for i in ann["segmentation"]]
        inf_coco.dataset["annotations"][idx]["segmentation"] = seg

    # save annotations
    if save_coco:
        if not save_coco.endswith("/"):
            save_coco = save_coco + "/"
        with open(save_coco + "detections_poly.json", "w") as json_file:
            json.dump(inf_coco.dataset, json_file, sort_keys=True, indent=4)

    return inf_coco


def convert_seg_poly(inf_coco: Type[COCO]) -> Type[COCO]:
    """converts segmentation from detectron2 format to polygons.

    Args:
      inf_coco : coco object with annotations dictionary containing detectron2 formatted segmentations

    Returns:
      inf_coco : coco object with annotations dictionary containing coco formatted polygons
    """

    def mask_to_polygons(mask):
        """from Detectron2 code"""
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res, hierarchy = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    for ann in tqdm(inf_coco.dataset["annotations"]):
        img = inf_coco.loadImgs([ann["image_id"]])
        height, width = img[0]["height"], img[0]["width"]
        segm = ann["segmentation"]

        if isinstance(segm["counts"], list):
            seg = maskUtils.frPyObjects(segm, height, width)
            mask = inf_coco.annToMask(seg)
        else:
            mask = inf_coco.annToMask(ann)

        mask = np.array(mask, dtype=np.uint8) * 255
        # we arrive at some error without two lines below, after these operations mask remains unchanged
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        ann["segmentation"], _ = mask_to_polygons(mask)

    return inf_coco
