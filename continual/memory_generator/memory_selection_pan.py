import os
import json
import copy
import tqdm
import numpy as np


def merge_prev_and_curr(prev_pan, prev_inst, curr_pan, curr_inst):
    merged_pan = copy.deepcopy(curr_pan)
    merged_inst = copy.deepcopy(curr_inst)
    for per_prev_img in prev_pan['images']:
        if per_prev_img['id'] not in [per_curr_img['id'] for per_curr_img in curr_pan['images']]:
            merged_pan['images'].append(per_prev_img)
    for per_prev_anno in prev_pan['annotations']:
        if per_prev_anno['image_id'] not in [per_curr_img['id'] for per_curr_img in curr_pan['images']]:
            merged_pan['annotations'].append(per_prev_anno)
        else:
            for per_curr_anno in merged_pan['annotations']:
                if per_curr_anno['image_id'] == per_prev_anno['image_id']:
                    per_curr_anno['segments_info'].extend(per_prev_anno['segments_info'])
                    break
    for per_prev_img in prev_inst['images']:
        if per_prev_img['id'] not in [per_curr_img['id'] for per_curr_img in curr_inst['images']]:
            merged_inst['images'].append(per_prev_img)
    for per_prev_anno in prev_inst['annotations']:
        merged_inst['annotations'].append(per_prev_anno)

    return merged_pan, merged_inst


def compute_global_nums(pan, global_segment_num=None):
    if global_segment_num is None:
        global_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in pan['categories']]}

    print("Computing global numbers...")
    for per_img_anno in tqdm.tqdm(pan['annotations']):
        segments_info = per_img_anno['segments_info']
        for seg in segments_info:
            global_segment_num[seg['category_id']] += 1

    target_segment_ratio = np.array(list(global_segment_num.values())) / np.sum(list(global_segment_num.values()))

    return target_segment_ratio, global_segment_num


def compute_stats(pan):
    segment_nums = {}

    print("Computing statistics...")
    for per_img_anno in tqdm.tqdm(pan['annotations']):
        per_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in pan['categories']]}

        segments_info = per_img_anno['segments_info']
        for seg in segments_info:
            per_segment_num[seg['category_id']] += 1
        segment_nums[per_img_anno['image_id']] = per_segment_num

    return segment_nums


def greedy_selection(
        images_data,
        num_categories,
        num_selections,
        target_segment_ratio,
        current_segment_num=None
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


def prepare_memory_cps(split, step):
    num_selections = 300
    if split == '100-10':
        base_cls = 100
        inc_cls = 10
        keep_ratios = [100 / 110, 110 / 120, 120 / 130, 130 / 140, 140 / 150]
    elif split == '100-50':
        base_cls = 100
        inc_cls = 50
        keep_ratios = [100 / 150]
    elif split == '100-5':
        base_cls = 100
        inc_cls = 5
        keep_ratios = [100 / 105, 105 / 110, 110 / 115, 115 / 120, 120 / 125,
                       125 / 130, 130 / 135, 135 / 140, 140 / 145, 145 / 150]
    elif split == '50-50':
        base_cls = 50
        inc_cls = 50
        keep_ratios = [50 / 100, 100 / 150]
    keep_ratio = 1 if step == 1 else keep_ratios[step - 2]

    pan_path = f"json/pan/train_{base_cls}-{inc_cls}_step{step}_pan.json"
    inst_path = f"json/pan/train_{base_cls}-{inc_cls}_step{step}_inst.json"
    output_pan_path = f"json/memory/pan/train_{base_cls}-{inc_cls}_step{step}_pan.json"
    output_inst_path = f"json/memory/pan/train_{base_cls}-{inc_cls}_step{step}_inst.json"
    output_target_path = f"json/memory/pan/train_{base_cls}-{inc_cls}_step{step}_target.json"
    with open(pan_path, 'r') as f:
        pan = json.load(f)
    with open(inst_path, 'r') as f:
        inst = json.load(f)
    os.makedirs("json/memory/pan", exist_ok=True)

    if step > 1:
        prev_pan_path = f"json/memory/pan/train_{base_cls}-{inc_cls}_step{step - 1}_pan.json"
        prev_inst_path = f"json/memory/pan/train_{base_cls}-{inc_cls}_step{step - 1}_inst.json"
        prev_target_path = f"json/memory/pan/train_{base_cls}-{inc_cls}_step{step - 1}_target.json"
        with open(prev_pan_path, 'r') as f:
            prev_pan = json.load(f)
        with open(prev_inst_path, 'r') as f:
            prev_inst = json.load(f)
        with open(prev_target_path, 'r') as f:
            prev_target = json.load(f)

        merged_pan, merged_inst = merge_prev_and_curr(prev_pan, prev_inst, pan, inst)

        prev_global_segment_nums = prev_target['global_segment_num']
        global_segment_num = \
            {int(cat_id): prev_global_segment_nums[cat_id] for cat_id in prev_global_segment_nums.keys()}

        target_segment_ratio, global_segment_num = compute_global_nums(pan, global_segment_num)
        segment_nums = compute_stats(merged_pan)

        prev_image_ids = [img['id'] for img in prev_pan['images']]
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

        pan = merged_pan
        inst = merged_inst

    else:
        target_segment_ratio, global_segment_num = compute_global_nums(pan)
        segment_nums = compute_stats(pan)

        images_data = {}
        for img_id in segment_nums.keys():
            images_data[img_id] = list(segment_nums[img_id].values())
        print("Selecting images...")
        selected_images, current_segment_num = \
            greedy_selection(
                images_data=images_data,
                num_categories=target_segment_ratio.shape[0],
                num_selections=num_selections,
                target_segment_ratio=target_segment_ratio,
            )

    new_pan = {
        'images': [],
        'annotations': [],
        'categories': pan['categories']
    }
    new_inst = {
        'images': [],
        'annotations': [],
        'categories': inst['categories']
    }

    for img in pan['images']:
        if img['id'] in selected_images:
            new_pan['images'].append(img)
    for ann in pan['annotations']:
        if ann['image_id'] in selected_images:
            new_pan['annotations'].append(ann)

    for img in inst['images']:
        if img['id'] in selected_images:
            new_inst['images'].append(img)
    for ann in inst['annotations']:
        if ann['image_id'] in selected_images:
            new_inst['annotations'].append(ann)

    with open(output_pan_path, 'w') as f:
        json.dump(new_pan, f)
    with open(output_inst_path, 'w') as f:
        json.dump(new_inst, f)

    stats = {
        'global_segment_num': global_segment_num,
    }
    with open(output_target_path, 'w') as f:
        json.dump(stats, f)


print("Preparing panoptic 100-10...")
for i in range(1, 6):
    print("Step", i)
    prepare_memory_cps(split='100-10', step=i)
print("Preparing panoptic 100-50...")
for i in range(1, 2):
    print("Step", i)
    prepare_memory_cps(split='100-50', step=i)
print("Preparing panoptic 100-5...")
for i in range(1, 11):
    print("Step", i)
    prepare_memory_cps(split='100-5', step=i)
print("Preparing panoptic 50-50...")
for i in range(1, 3):
    print("Step", i)
    prepare_memory_cps(split='50-50', step=i)
