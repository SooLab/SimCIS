import os
import copy
import json
import tqdm
import numpy as np


def merge_prev_and_curr(prev_inst, curr_inst):
    merged_inst = copy.deepcopy(curr_inst)
    curr_img_list = [per_curr_img['id'] for per_curr_img in curr_inst['images']]
    for per_prev_img in prev_inst['images']:
        if per_prev_img['id'] not in curr_img_list:
            merged_inst['images'].append(per_prev_img)
    for per_prev_anno in prev_inst['annotations']:
        merged_inst['annotations'].append(per_prev_anno)

    return merged_inst


def compute_global_nums(inst, global_segment_num=None):
    if global_segment_num is None:
        global_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in inst['categories']]}

    print("Computing global numbers...")
    cur_img = inst['annotations'][0]['image_id']
    for per_anno in tqdm.tqdm(inst['annotations']):
        if per_anno['image_id'] != cur_img:
            cur_img = per_anno['image_id']
        global_segment_num[per_anno['category_id']] += 1

    target_segment_ratio = np.array(list(global_segment_num.values())) / np.sum(list(global_segment_num.values()))

    return (target_segment_ratio, global_segment_num)


def compute_stats(inst):
    segment_nums = {}

    print("Computing statistics...")
    for per_anno in tqdm.tqdm(inst['annotations']):
        if per_anno['image_id'] not in segment_nums.keys():
            per_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in inst['categories']]}
        else:
            per_segment_num = segment_nums[per_anno['image_id']]
        per_segment_num[per_anno['category_id']] += 1

        segment_nums[per_anno['image_id']] = per_segment_num

    return segment_nums


def greedy_selection(
        images_data,
        num_categories,
        num_selections,
        target_segment_ratio,
        current_segment_num=None,
):
    selected_images = []
    if current_segment_num is None:
        current_segment_num = np.zeros(num_categories)

    for _ in tqdm.tqdm(range(num_selections)):
        best_img = None
        best_score = float('inf')

        for img, segment_num in images_data.items():
            if img in selected_images:
                continue

            new_segment_num = (current_segment_num + np.array(segment_num))
            new_segment_ratio = new_segment_num / np.sum(new_segment_num)

            segment_diff = np.sum(np.abs(new_segment_ratio - target_segment_ratio))

            if segment_diff < best_score:
                best_score = segment_diff
                best_img = img

        selected_images.append(best_img)
        current_segment_num = current_segment_num + np.array(images_data[best_img])

    return selected_images, current_segment_num


def prepare_memory_cis(split, step):
    num_selections = 300
    tot_cls = 100
    if split == '50-10':
        base_cls = 50
        inc_cls = 10
        keep_ratios = [50 / 60, 60 / 70, 70 / 80, 80 / 90, 90 / 100]
    elif split == '50-50':
        base_cls = 50
        inc_cls = 50
        keep_ratios = [50 / 100]
    elif split == '50-5':
        base_cls = 50
        inc_cls = 5
        keep_ratios = [50 / 55, 55 / 60, 60 / 65, 65 / 70, 70 / 75, 75 / 80, 80 / 85, 85 / 90, 90 / 95, 95 / 100]
    keep_ratio = 1 if step == 1 else keep_ratios[step - 2]

    inst_path = f"json/inst/train_{base_cls}-{inc_cls}_step{step}_inst.json".format(step)
    output_inst_path = f"json/memory/inst/train_{base_cls}-{inc_cls}_step{step}_inst.json".format(step)
    output_target_path = f"json/memory/inst/train_{base_cls}-{inc_cls}_step{step}_target.json".format(step)
    with open(inst_path, 'r') as f:
        inst = json.load(f)
    os.makedirs("json/memory/inst", exist_ok=True)

    if step > 1:
        prev_inst_path = f"json/memory/inst/train_{base_cls}-{inc_cls}_step{step - 1}_inst.json"
        prev_target_path = f"json/memory/inst/train_{base_cls}-{inc_cls}_step{step - 1}_target.json"
        with open(prev_inst_path, 'r') as f:
            prev_inst = json.load(f)
        with open(prev_target_path, 'r') as f:
            prev_target = json.load(f)

        merged_inst = merge_prev_and_curr(prev_inst, inst)

        prev_global_segment_nums = prev_target['global_segment_num']
        global_segment_num = \
            {int(cat_id): prev_global_segment_nums[cat_id] for cat_id in prev_global_segment_nums.keys()}

        target_segment_ratio, global_segment_num = compute_global_nums(inst, global_segment_num)
        segment_nums = compute_stats(merged_inst)

        prev_image_ids = [img['id'] for img in prev_inst['images']]
        prev_images_data = {}
        for img_id in prev_image_ids:
            prev_images_data[img_id] = list(segment_nums[img_id].values())
        print("Selecting previous images...")
        selected_prev_images, current_segment_num = greedy_selection(
            images_data=prev_images_data,
            num_categories=target_segment_ratio.shape[0],
            num_selections=int(num_selections * keep_ratio),
            target_segment_ratio=target_segment_ratio,
        )

        images_data = {}
        for img_id in list(set(segment_nums.keys()) - set(prev_image_ids)):
            images_data[img_id] = list(segment_nums[img_id].values())

        print("Selecting images...")
        selected_curr_images, current_segment_num = greedy_selection(
            images_data=images_data,
            num_categories=target_segment_ratio.shape[0],
            num_selections=num_selections - int(num_selections * keep_ratio),
            target_segment_ratio=target_segment_ratio,
            current_segment_num=current_segment_num,
        )
        selected_images = selected_prev_images + selected_curr_images

        inst = merged_inst

    else:
        target_segment_ratio, global_segment_num = compute_global_nums(inst)
        segment_nums = compute_stats(inst)

        images_data = {}
        for img_id in segment_nums.keys():
            images_data[img_id] = list(segment_nums[img_id].values())
        print("Selecting images...")

        selected_images, current_segment_num = greedy_selection(
            images_data=images_data,
            num_categories=target_segment_ratio.shape[0],
            num_selections=num_selections,
            target_segment_ratio=target_segment_ratio
        )

    new_inst = {
        'images': [],
        'annotations': [],
        'categories': inst['categories']
    }
    for img in inst['images']:
        if img['id'] in selected_images:
            new_inst['images'].append(img)
    for ann in inst['annotations']:
        if ann['image_id'] in selected_images and segment_nums[ann['image_id']][ann['category_id']]:
            new_inst['annotations'].append(ann)

    with open(output_inst_path, 'w') as f:
        json.dump(new_inst, f)

    stats = {
        'global_segment_num': global_segment_num
    }
    with open(output_target_path, 'w') as f:
        json.dump(stats, f)


print("Preparing instance 50-10...")
for i in range(1, 6):
    print("Step", i)
    prepare_memory_cis(split='50-10', step=i)
print("Preparing instance 50-50...")
for i in range(1, 2):
    print("Step", i)
    prepare_memory_cis(split='50-50', step=i)
print("Preparing instance 50-5...")
for i in range(1, 11):
    print("Step", i)
    prepare_memory_cis(split='50-5', step=i)

