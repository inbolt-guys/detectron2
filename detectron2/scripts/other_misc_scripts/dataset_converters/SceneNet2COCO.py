from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import coco_utils as coco_utils
import json
import pycocotools.mask as mask_util


def save_coco(images_annotations, annotations, category_ids, output_folder, dataset):
    """
        images, list, create_image_annotation(file, w, h, id)
        annotations, list, create_annotation_format(imageid, category, annotationid, mask)
        category_ids, dict, category_name : category_id
    )
    """
    coco_format = coco_utils.get_coco_json_format()
    coco_format["categories"] = coco_utils.create_category_annotation(category_ids)
    coco_format["images"] = images_annotations
    coco_format["annotations"] = annotations
    with open(os.path.join(output_folder, f"annotations_{dataset}.json"), "w") as outfile:
        json.dump(coco_format, outfile)


datasets = ["train", "test"]
out_dir = f"/home/inbolt/data_hdd/R&D/clara/datasets/SceneNet/SceneNet_COCO"


for d in datasets:
    image_paths = sorted(
        glob.glob(f"/home/inbolt/data_hdd/R&D/clara/datasets/SceneNet/{d}/*/*/photo/*.jpg")
    )

    images_annotations = []
    annotations = []
    category_ids = {"object": 1}
    image_id = 0
    annotation_id = 0
    i = 0

    dataset_output_dir = out_dir + "/" + d + str(i)
    os.makedirs(dataset_output_dir, exist_ok=True)

    for ip in image_paths:
        image_id += 1

        folder = os.path.dirname(ip)
        parent_folder = os.path.dirname(folder)
        parent_parent_folder = os.path.dirname(parent_folder)
        im_name = os.path.basename(ip)

        rgb = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(
            os.path.join(parent_folder, "depth", im_name).replace("jpg", "png"),
            cv2.IMREAD_UNCHANGED,
        )
        segm = cv2.imread(
            os.path.join(parent_folder, "instance", im_name).replace("jpg", "png"),
            cv2.IMREAD_UNCHANGED,
        )

        rgbd = np.dstack((rgb.astype(np.uint16), depth.astype(np.uint16)))
        rgbd_name = (
            os.path.basename(os.path.dirname(parent_parent_folder))
            + "_"
            + os.path.basename(parent_folder)
            + "_"
            + im_name.replace("jpg", "png")
        )

        cv2.imwrite(dataset_output_dir + "/" + rgbd_name, rgbd)
        h, w = rgb.shape[:2]
        images_annotations.append(coco_utils.create_image_annotation(rgbd_name, w, h, image_id))

        per_object_masks = [segm == i for i in range(2, np.max(segm) + 1) if np.any(segm == i)]
        for object in per_object_masks:
            annotation_id += 1
            annotations.append(
                coco_utils.create_annotation_format(
                    image_id,
                    1,
                    annotation_id,
                    encoded_mask=mask_util.encode(np.asfortranarray(object)),
                )
            )

        if image_id > 1_000_000:
            save_coco(images_annotations, annotations, category_ids, out_dir, d + str(i))
            images_annotations = []
            annotations = []
            category_ids = {"object": 1}
            image_id = 0
            annotation_id = 0
            i += 1
            dataset_output_dir = out_dir + "/" + d + str(i)
            os.makedirs(dataset_output_dir, exist_ok=True)

    save_coco(images_annotations, annotations, category_ids, out_dir, d + str(i))
