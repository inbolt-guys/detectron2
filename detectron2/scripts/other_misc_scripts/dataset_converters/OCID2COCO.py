from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import coco_utils
import json
import pycocotools.mask as mask_util
import random


def save_coco(images_annotations, annotations, category_ids, output_folder):
    """
        images, list, create_image_annotation(file, w, h, id)
        annotations, list, create_annotation_format(imageid, category, annotationid, mask)
        category_ids, dict, category_name : category_id
    )
    """
    print(len(image_paths))
    for k in images_annotations:
        print(k, len(images_annotations[k]))
        coco_format = coco_utils.get_coco_json_format()
        coco_format["categories"] = coco_utils.create_category_annotation(category_ids)
        coco_format["images"] = images_annotations[k]
        coco_format["annotations"] = annotations[k]
        with open(os.path.join(output_folder, f"annotations_{k}.json"), "w") as outfile:
            json.dump(coco_format, outfile)


datasets = ["ARID10", "ARID20", "YCB10"]
out_dir = f"/home/inbolt/data_hdd/R&D/clara/datasets/OCID-dataset/OCID_COCO"
for i in ["test", "train"]:
    dataset_output_dir = os.path.join(out_dir, i)
    os.makedirs(dataset_output_dir, exist_ok=True)

images_annotations = {"train": [], "test": []}
annotations = {"train": [], "test": []}
category_ids = {"object": 1}
image_id = {"train": 0, "test": 0}
annotation_id = {"train": 0, "test": 0}
image_paths = []

for d in datasets:
    patterns = [
        f"/home/inbolt/data_hdd/R&D/clara/datasets/OCID-dataset/{d}/*/*/*/*/rgb/*.png",
        f"/home/inbolt/data_hdd/R&D/clara/datasets/OCID-dataset/{d}/*/*/*/*/*/rgb/*.png",
    ]
    for p in patterns:
        image_paths.extend(sorted(glob.glob(p)))

for ip in image_paths:

    folder = os.path.dirname(ip)
    parent_folder = os.path.dirname(folder)
    im_name = os.path.basename(ip)

    seqn = int(os.path.basename(parent_folder).replace("seq", ""))
    is_train = seqn % 6 > 0
    if is_train:
        image_id["train"] += 1
        dataset_output_dir = os.path.join(out_dir, "train")
    else:
        image_id["test"] += 1
        dataset_output_dir = os.path.join(out_dir, "test")

    rgb = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(os.path.join(parent_folder, "depth", im_name), cv2.IMREAD_UNCHANGED)
    segm = cv2.imread(os.path.join(parent_folder, "label", im_name), cv2.IMREAD_UNCHANGED)

    rgbd = np.dstack((rgb.astype(np.uint16), depth.astype(np.uint16)))

    cv2.imwrite(dataset_output_dir + "/" + im_name, rgbd)
    h, w = rgb.shape[:2]
    if is_train:
        images_annotations["train"].append(
            coco_utils.create_image_annotation(im_name, w, h, image_id["train"])
        )
    else:
        images_annotations["test"].append(
            coco_utils.create_image_annotation(im_name, w, h, image_id["test"])
        )

    per_object_masks = [segm == i for i in range(2, np.max(segm) + 1) if np.any(segm == i)]
    if is_train:
        for object in per_object_masks:
            annotation_id["train"] += 1
            annotations["train"].append(
                coco_utils.create_annotation_format(
                    image_id["train"],
                    1,
                    annotation_id["train"],
                    encoded_mask=mask_util.encode(np.asfortranarray(object)),
                )
            )
    else:
        for object in per_object_masks:
            annotation_id["test"] += 1
            annotations["test"].append(
                coco_utils.create_annotation_format(
                    image_id["test"],
                    1,
                    annotation_id["test"],
                    encoded_mask=mask_util.encode(np.asfortranarray(object)),
                )
            )

save_coco(images_annotations, annotations, category_ids, out_dir)
