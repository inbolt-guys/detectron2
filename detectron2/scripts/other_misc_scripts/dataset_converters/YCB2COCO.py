from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import coco_utils as coco_utils
import json
import pycocotools.mask as mask_util


def save_coco(images_annotations, annotations, category_ids, output_folder, d):
    """
        images, list, create_image_annotation(file, w, h, id)
        annotations, list, create_annotation_format(imageid, category, annotationid, mask)
        category_ids, dict, category_name : category_id
    )
    """
    for k in images_annotations:
        coco_format = coco_utils.get_coco_json_format()
        coco_format["categories"] = coco_utils.create_category_annotation(category_ids)
        coco_format["images"] = images_annotations[k]
        coco_format["annotations"] = annotations[k]
        with open(os.path.join(output_folder, f"annotations_{d}_{k}.json"), "w") as outfile:
            json.dump(coco_format, outfile)


out_dir = f"/home/inbolt/data_hdd/R&D/clara/datasets/YCB/YCB_COCO"

for d in ["data"]:

    images_annotations = {"train": [], "test": []}
    annotations = {"train": [], "test": []}
    category_ids = {"object": 1}
    image_id = {"train": 0, "test": 0}
    annotation_id = {"train": 0, "test": 0}
    for i in [d + "_test", d + "_train"]:
        dataset_output_dir = os.path.join(out_dir, i)
        os.makedirs(dataset_output_dir, exist_ok=True)

    if d == "data":
        image_paths = sorted(
            glob.glob(f"/home/inbolt/data_hdd/R&D/clara/datasets/YCB/{d}/00*/*-color.jpg")
        )
    else:
        image_paths = sorted(
            glob.glob(f"/home/inbolt/data_hdd/R&D/clara/datasets/YCB/{d}/*-color.jpg")
        )

    for ip in image_paths:
        folder = os.path.dirname(ip)
        im_name = os.path.basename(ip)
        number = im_name.replace("-color.jpg", "")

        if d == "data":
            seqn = int(os.path.basename(folder))
            is_train = seqn % 6 > 0
        else:
            is_train = int(number) % 6 > 0
        if is_train:
            image_id["train"] += 1
            dataset_output_dir = os.path.join(out_dir, d + "_train")
        else:
            image_id["test"] += 1
            dataset_output_dir = os.path.join(out_dir, d + "_test")

        rgb = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(os.path.join(folder, number + "-depth.png"), cv2.IMREAD_UNCHANGED)
        segm = np.array(Image.open(os.path.join(folder, number + "-label.png")))

        rgbd = np.dstack((rgb.astype(np.uint16), depth.astype(np.uint16)))
        if d == "data":
            rgbd_name = "rgbd_" + os.path.basename(folder) + "_" + number + ".png"
        else:
            rgbd_name = "rgbd_" + number + ".png"

        if cv2.imread(dataset_output_dir + "/" + rgbd_name, cv2.IMREAD_UNCHANGED) is None:
            cv2.imwrite(dataset_output_dir + "/" + rgbd_name, rgbd)
        h, w = rgb.shape[:2]
        if is_train:
            images_annotations["train"].append(
                coco_utils.create_image_annotation(rgbd_name, w, h, image_id["train"])
            )
        else:
            images_annotations["test"].append(
                coco_utils.create_image_annotation(rgbd_name, w, h, image_id["test"])
            )

        per_object_masks = [segm == i for i in range(1, np.max(segm) + 1) if np.any(segm == i)]
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

    save_coco(images_annotations, annotations, category_ids, out_dir, d)
