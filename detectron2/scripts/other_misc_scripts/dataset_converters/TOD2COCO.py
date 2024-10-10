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


datasets = ["training_set", "test_set"]
out_dir = f"/home/inbolt/data_hdd/R&D/clara/datasets/TOD/TOD_COCO"


for d in datasets:
    image_paths = sorted(
        glob.glob(f"/home/inbolt/data_hdd/R&D/clara/datasets/TOD/{d}/scene_*/rgb_*")
    )
    dataset_output_dir = out_dir + "/" + d
    os.makedirs(dataset_output_dir, exist_ok=True)

    images_annotations = []
    annotations = []
    category_ids = {"object": 1}
    image_id = 0
    annotation_id = 0

    for ip in image_paths:
        image_id += 1
        folder = os.path.dirname(ip)
        im_name = os.path.basename(ip)
        number = im_name.replace(".jpeg", "").replace("rgb_", "")

        rgb = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(folder + "/depth_" + str(number) + ".png", cv2.IMREAD_UNCHANGED)
        segm = np.array(Image.open(folder + "/segmentation_" + str(number) + ".png"))

        rgbd = np.dstack((rgb.astype(np.uint16), depth.astype(np.uint16)))
        rgbd_name = "rgbd_" + os.path.basename(folder).replace("scene_", "") + "_" + number + ".png"

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

    save_coco(images_annotations, annotations, category_ids, out_dir, d)
