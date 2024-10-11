import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from skimage import measure
import cv2
import pycocotools.mask as mask_util


def create_sub_masks(mask_image, width, height):
    """
    Create a list of binary masks, one for each object in the image.

    Arguments:
        mask_image: a binary mask image of size [height, width].
        width: the width of the image.
        height: the height of the image.

    Returns:
        A list of binary masks, one for each object in the image.
    """

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))

            # Check to see if we have created a sub-mask...
            if pixel != 0:  # not a background pixel
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn"t handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new("1", (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask):
    """
    Create a polygon for an object composed of several sub-masks.

    Arguments:
        sub_mask: a single binary mask composed of multiple sub-masks.

    Returns:
        A polygon for the object.
    """

    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if poly.is_empty:
            # Go to next iteration, dont save empty values in list
            continue

        try:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
            polygons.append(poly)
        except AttributeError:
            continue
    return polygons, segmentations


def create_category_annotation(category_dict):
    """
    Create category annotation.

    Arguments:
        category_dict: a dictionary of categories indexed by category_id.

    Returns:
        A list of categories.
    """
    category_list = []

    for key, value in category_dict.items():
        category = {"supercategory": key, "id": value, "name": key}
        category_list.append(category)

    return category_list


def create_image_annotation(file_name, width, height, image_id):
    """
    Create image annotation.

    Arguments:
        file_name: name of the image.
        width: width of the image.
        height: height of the image.
        image_id: id of the image.

    Returns:
        A list of images.
    """
    images = {"file_name": file_name, "height": height, "width": width, "id": image_id}

    return images


def crop_coco_bbox(bbox, image_height, image_width):
    box_min_x, box_min_y, box_width, box_height = bbox[0], bbox[1], bbox[2], bbox[3]
    if box_min_x + box_width > image_width:
        box_width = image_width - box_min_x
    if box_min_y + box_height > image_height:
        box_height = image_height - box_min_y
    return [box_min_x, box_min_y, box_width, box_height]


def create_annotation_format(
    image_id, category_id, annotation_id, polygon=None, segmentation=None, encoded_mask=None
):
    """
    Create annotation format required by COCO.

    Arguments:

        polygon: a list of polygons.
        segmentation: a list of segmentations.
        image_id: id of the image.
        category_id: id of the category.
        annotation_id: id of the annotation.

    Returns:
        annotation: annotation dict in COCO format.
    """
    if encoded_mask is None:
        min_x, min_y, max_x, max_y = polygon.bounds
        width = max_x - min_x
        height = max_y - min_y
        bbox = (min_x, min_y, width, height)
        area = polygon.area
    else:
        area = int(mask_util.area(encoded_mask))
        bbox = mask_util.toBbox(encoded_mask).tolist()
        segmentation = encoded_mask
        segmentation["counts"] = segmentation["counts"].decode("utf-8")

    bbox

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id,
    }

    return annotation


def get_coco_json_format():
    """
    Create coco json format.

    Arguments:
        None

    Returns:
        coco_format: a coco json format."""
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}],
    }

    return coco_format


def get_points(poly):
    """
    Get points from polygon.

    Arguments:
        poly: a polygon.

    Returns:
        xx: x coordinates of polygon.
        yy: y coordinates of polygon."""
    xx, yy = poly.exterior.coords.xy
    xx.append(xx[0])
    yy.append(yy[0])
    return (xx, yy)


def normalize_depth(
    depth,
    min_depth=None,
    max_depth=None,
    reverse=True,
    percentileForMin=5,
    percentileForMax=85,
    scaleFactorForMin=1 / 1.5,
    valueWhenLessThanMin=255,
    valueWhenMoreThanMax=255,
    valueWhenNoValue=255,
):
    noValueMask = depth == 0
    if not max_depth:
        max_depth = np.percentile(depth[np.logical_not(noValueMask)], percentileForMax)
    if not min_depth:
        min_depth = (
            np.percentile(depth[np.logical_not(noValueMask)], percentileForMin) * scaleFactorForMin
        )

    depth_normalized = np.copy(depth)

    lessThanMinMask = np.logical_and(depth_normalized < min_depth, np.logical_not(noValueMask))
    moreThanMaxMask = depth_normalized > max_depth

    depth_normalized = (depth_normalized - min_depth) / (max_depth - min_depth) * 255

    depth_normalized[noValueMask] = valueWhenNoValue
    depth_normalized[lessThanMinMask] = valueWhenLessThanMin
    depth_normalized[moreThanMaxMask] = valueWhenMoreThanMax

    if reverse:
        depth_normalized = 255 - depth_normalized

    return depth_normalized.astype(np.uint8)


def hole_fill_depth(depth_map):
    """
    Fill holes in a depth map using OpenCV's inpainting method.

    Parameters:
    depth_map (numpy.ndarray): Input depth map with holes (zeros).

    Returns:
    numpy.ndarray: Depth map with filled holes in uint16 format.
    """
    # Create a mask of the holes (where depth value is 0)
    mask = (depth_map == 0).astype(np.uint8) * 255

    # Convert the depth map to float32 for inpainting
    depth_map_float = depth_map.astype(np.float32)

    # Use OpenCV's inpainting function to fill the holes
    inpainted_depth_map_float = cv2.inpaint(
        depth_map_float, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    # Convert the inpainted depth map back to uint16
    inpainted_depth_map = inpainted_depth_map_float.astype(np.uint16)

    return inpainted_depth_map
