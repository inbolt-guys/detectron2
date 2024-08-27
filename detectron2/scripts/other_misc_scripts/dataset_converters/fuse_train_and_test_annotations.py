import json
import copy
import os 

def fuse_coco_annotations(file1_path, file2_path, output_path):
    # Load the two annotation files
    with open(file1_path, 'r') as f:
        coco1 = json.load(f)
    
    with open(file2_path, 'r') as f:
        coco2 = json.load(f)
    
    # Create a deep copy of the first COCO dataset to be the base
    fused_coco = copy.deepcopy(coco1)
    
    # Offset for the second dataset's IDs (to avoid ID conflicts)
    image_id_offset = max([img['id'] for img in coco1['images']])
    annotation_id_offset = max([ann['id'] for ann in coco1['annotations']])
    
    # Fuse images
    for img in coco2['images']:
        img_copy = copy.deepcopy(img)
        img_copy['id'] += image_id_offset
        fused_coco['images'].append(img_copy)
    
    # Fuse annotations
    for ann in coco2['annotations']:
        ann_copy = copy.deepcopy(ann)
        ann_copy['id'] += annotation_id_offset
        ann_copy['image_id'] += image_id_offset
        fused_coco['annotations'].append(ann_copy)
    
    # Fuse categories (assuming both files have the same categories)
    # If categories differ, you'll need to handle this separately.
    
    # Save the fused annotations to a new file
    with open(output_path, 'w') as f:
        json.dump(fused_coco, f, indent=4)
    
    print(f"Fused annotation file saved to {output_path}")

dataset_path= "/home/clara/detectronDocker/dataset_for_detectron/OCID_COCO"
fuse_coco_annotations(os.path.join(dataset_path, "annotations_train.json"), os.path.join(dataset_path, "annotations_test.json"), os.path.join(dataset_path, "annotations_all.json"))