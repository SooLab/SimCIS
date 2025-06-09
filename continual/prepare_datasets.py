import json
import os


_ROOT = "datasets"
_TRAIN_JSON_PAN = "ADEChallengeData2016/ade20k_panoptic_train.json"
_VAL_JSON_PAN = "ADEChallengeData2016/ade20k_panoptic_val.json"
_TRAIN_JSON_INST = "ADEChallengeData2016/ade20k_instance_train.json"
_VAL_JSON_INST = "ADEChallengeData2016/ade20k_instance_val.json"
_OUTPUT_DIR_PAN = "json/pan"
_OUTPUT_DIR_INST = "json/inst"


def modify_json_pan(json_file, output_path, category_ids_to_keep):
    # if os.path.exists(output_path):
    #     print(f"File {output_path} already exists. Skipping...")
    #     return

    with open(json_file, 'r') as f:
        dataset = json.load(f)

    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories']

    filtered_images = []
    filtered_annotations = []
    for image, annotation in zip(images, annotations):
        segments_info = annotation['segments_info']
        filtered_segments_info = []
        for segment_info in segments_info:
            category_id = segment_info['category_id']
            if category_id in category_ids_to_keep:
                filtered_segments_info.append(segment_info)
        if len(filtered_segments_info) > 0:
            annotation['segments_info'] = filtered_segments_info
            filtered_images.append(image)
            filtered_annotations.append(annotation)

    dataset['images'] = filtered_images
    dataset['annotations'] = filtered_annotations
    dataset['categories'] = categories

    with open(output_path, 'w') as f:
        print(f"Writing to {output_path}")
        json.dump(dataset, f)


def modify_json_inst(json_file, output_path, category_ids_to_keep, image_list=None):
    # if os.path.exists(output_path):
    #     print(f"File {output_path} already exists. Skipping...")
    #     return

    with open(json_file, 'r') as f:
        dataset = json.load(f)

    annotations = dataset['annotations']
    filtered_annotations = []
    for annotation in annotations:
        if annotation['category_id'] in category_ids_to_keep:
            filtered_annotations.append(annotation)

    filtered_images = []
    if image_list is None:
        image_list = list(set([annotation['image_id'] for annotation in filtered_annotations]))
    for image in dataset['images']:
        if image['id'] in image_list:
            filtered_images.append(image)

    dataset['images'] = filtered_images
    dataset['annotations'] = filtered_annotations

    with open(output_path, 'w') as f:
        print(f"Writing to {output_path}")
        json.dump(dataset, f)


def image_filter_pan(tot_cls, base_cls, inc_cls, task):
    if not os.path.exists(_OUTPUT_DIR_PAN):
        os.makedirs(_OUTPUT_DIR_PAN)

    num_tasks = 1 + (tot_cls - base_cls) // inc_cls
    n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)
    category_ids_to_keep_train = list(range(sum(n_cls_in_tasks[:task - 1]), sum(n_cls_in_tasks[:task])))
    category_ids_to_keep_val = list(range(sum(n_cls_in_tasks[:task])))

    train_json_file_pan = os.path.join(_ROOT, _TRAIN_JSON_PAN)
    val_json_file_pan = os.path.join(_ROOT, _VAL_JSON_PAN)

    train_json_file_inst = os.path.join(_ROOT, _TRAIN_JSON_INST)
    val_json_file_inst = os.path.join(_ROOT, _VAL_JSON_INST)

    modify_json_pan(
        train_json_file_pan,
        os.path.join(_OUTPUT_DIR_PAN, f"train_{base_cls}-{inc_cls}_step{task}_pan.json"),
        category_ids_to_keep_train
    )
    modify_json_pan(
        val_json_file_pan,
        os.path.join(_OUTPUT_DIR_PAN, f"val_{base_cls}-{inc_cls}_step{task}_pan.json"),
        category_ids_to_keep_val
    )

    with open(os.path.join(_OUTPUT_DIR_PAN, f"train_{base_cls}-{inc_cls}_step{task}_pan.json"), 'r') as f:
        pan_train = json.load(f)
    with open(os.path.join(_OUTPUT_DIR_PAN, f"val_{base_cls}-{inc_cls}_step{task}_pan.json"), 'r') as f:
        pan_val = json.load(f)
    image_list_train = list(set([img['id'] for img in pan_train['images']]))
    image_list_val = list(set([img['id'] for img in pan_val['images']]))

    modify_json_inst(
        train_json_file_inst,
        os.path.join(_OUTPUT_DIR_PAN, f"train_{base_cls}-{inc_cls}_step{task}_inst.json"),
        category_ids_to_keep_train,
        image_list=image_list_train
    )
    modify_json_inst(
        val_json_file_inst,
        os.path.join(_OUTPUT_DIR_PAN, f"val_{base_cls}-{inc_cls}_step{task}_inst.json"),
        category_ids_to_keep_val,
        image_list=image_list_val
    )


def image_filter_inst(tot_cls, base_cls, inc_cls, task):
    if not os.path.exists(_OUTPUT_DIR_INST):
        os.makedirs(_OUTPUT_DIR_INST)

    inst_ids = [7, 8, 10, 12, 14, 15, 18, 19, 20, 22, 23, 24, 27, 30, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44,
                45, 47, 49, 50, 53, 55, 56, 57, 58, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 82,
                83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 97, 98, 102, 103, 104, 107, 108, 110, 111, 112, 115, 116, 118,
                119, 120, 121, 123, 124, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144,
                146, 147, 148, 149]

    num_tasks = 1 + (tot_cls - base_cls) // inc_cls
    n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)
    category_ids_to_keep_train = inst_ids[sum(n_cls_in_tasks[:task - 1]): sum(n_cls_in_tasks[:task])]
    category_ids_to_keep_val = inst_ids[:sum(n_cls_in_tasks[:task])]

    train_json_file_inst = os.path.join(_ROOT, _TRAIN_JSON_INST)
    val_json_file_inst = os.path.join(_ROOT, _VAL_JSON_INST)

    modify_json_inst(
        train_json_file_inst,
        os.path.join(_OUTPUT_DIR_INST, f"train_{base_cls}-{inc_cls}_step{task}_inst.json"),
        category_ids_to_keep_train
    )
    modify_json_inst(
        val_json_file_inst,
        os.path.join(_OUTPUT_DIR_INST, f"val_{base_cls}-{inc_cls}_step{task}_inst.json"),
        category_ids_to_keep_val
    )

if __name__ == "__main__":
    print("Preparing panoptic 100-10...")
    for i in range(1, 7):
        image_filter_pan(tot_cls=150, base_cls=100, inc_cls=10, task=i)
    print("Preparing panoptic 100-50...")
    for i in range(1, 3):
        image_filter_pan(tot_cls=150, base_cls=100, inc_cls=50, task=i)
    print("Preparing panoptic 100-5...")
    for i in range(1, 12):
        image_filter_pan(tot_cls=150, base_cls=100, inc_cls=5, task=i)
    print("Preparing panoptic 50-50...")
    for i in range(1, 4):
        image_filter_pan(tot_cls=150, base_cls=50, inc_cls=50, task=i)

    print("Preparing instance 50-10...")
    for i in range(1, 7):
        image_filter_inst(tot_cls=100, base_cls=50, inc_cls=10, task=i)
    print("Preparing instance 50-50...")
    for i in range(1, 3):
        image_filter_inst(tot_cls=100, base_cls=50, inc_cls=50, task=i)
    print("Preparing instance 50-5...")
    for i in range(1, 12):
        image_filter_inst(tot_cls=100, base_cls=50, inc_cls=5, task=i)
