import os
import json
import tqdm
import numpy as np
import skimage.io as io


def compute_global_nums(pan, sem_anno_dir, category_ids_to_keep, global_segment_num=None):
    if global_segment_num is None:
        global_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in pan['categories']]}

    print("Computing global numbers...")
    for per_img in tqdm.tqdm(pan['images']):
        per_img_anno_path = sem_anno_dir + per_img['id'] + ".png"
        per_img_anno = io.imread(per_img_anno_path)
        for cls in list(set(np.unique(per_img_anno)) & set(category_ids_to_keep)):
            global_segment_num[cls] += 1

    target_segment_ratio = np.array(list(global_segment_num.values())) / np.sum(list(global_segment_num.values()))

    return target_segment_ratio, global_segment_num


def compute_stats(pan, sem_anno_dir, category_ids_to_keep):
    segment_nums = {}

    print("Computing statistics...")
    for per_img in tqdm.tqdm(pan['images']):
        per_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in pan['categories']]}

        per_img_anno_path = sem_anno_dir + per_img['id'] + ".png"
        per_img_anno = io.imread(per_img_anno_path)
        for cls in list(set(np.unique(per_img_anno)) & set(category_ids_to_keep)):
            per_segment_num[cls] += 1
        segment_nums[per_img['id']] = per_segment_num

    return segment_nums


def update_stats_in_memory(prev_sem_anno_dir, category_ids_to_keep_past, segment_nums, pan):
    if segment_nums is None:
        segment_nums = {}

    print("Updating statistics in memory...")
    for per_img in tqdm.tqdm(os.listdir(prev_sem_anno_dir)):
        if per_img[:-4] not in segment_nums.keys():
            per_segment_num = {cat_id: 0 for cat_id in [cat['id'] for cat in pan['categories']]}
        else:
            per_segment_num = segment_nums[per_img[:-4]]

        per_img_anno_path = prev_sem_anno_dir + per_img
        per_img_anno = io.imread(per_img_anno_path)
        for cls in list(set(np.unique(per_img_anno)) & set(category_ids_to_keep_past)):
            per_segment_num[cls] += 1
        segment_nums[per_img[:-4]] = per_segment_num

    return segment_nums


def greedy_selection(images_data, num_categories, num_selections, target_segment_ratio, current_segment_num=None):
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


def generate_sem_memory_annotations(selected_images, segment_nums, sem_anno_dir, output_sem_anno_dir):
    print("Generating semantic memory annotations...")
    for img_id in tqdm.tqdm(selected_images):
        per_img_anno_path = sem_anno_dir + img_id + ".png"
        per_img_anno = io.imread(per_img_anno_path)
        for cls in np.unique(per_img_anno):
            if cls == 255:
                continue
            if segment_nums[img_id][cls] == 0:
                per_img_anno[per_img_anno == cls] = 255
        io.imsave(output_sem_anno_dir + img_id + ".png", per_img_anno)


def prepare_memory_css(split, step):
    num_selections = 300
    tot_cls = 150
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

    num_tasks = 1 + (tot_cls - base_cls) // inc_cls
    n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)
    category_ids_to_keep = list(range(sum(n_cls_in_tasks[:step - 1]), sum(n_cls_in_tasks[:step])))
    category_ids_to_keep_past = list(range(sum(n_cls_in_tasks[:step])))

    sem_anno_dir = "datasets/ADEChallengeData2016/annotations_detectron2/training/"
    pan_path = f"json/pan/train_{base_cls}-{inc_cls}_step{step}_pan.json"
    output_target_path = f"json/memory/sem/train_{base_cls}-{inc_cls}_step{step}_target.json"
    output_sem_anno_dir = f"json/memory/sem/train_{base_cls}-{inc_cls}_step{step}/"
    if not os.path.exists(output_sem_anno_dir):
        os.makedirs(output_sem_anno_dir)

    with open(pan_path, 'r') as f:
        pan = json.load(f)

    if step > 1:
        prev_sem_anno_dir = f"json/memory/sem/train_{base_cls}-{inc_cls}_step{step - 1}/"
        prev_target_path = f"json/memory/sem/train_{base_cls}-{inc_cls}_step{step - 1}_target.json"
        with open(prev_target_path, 'r') as f:
            prev_target = json.load(f)

        prev_global_segment_nums = prev_target['global_segment_num']
        global_segment_num = \
            {int(cat_id): prev_global_segment_nums[cat_id] for cat_id in prev_global_segment_nums.keys()}

        target_segment_ratio, global_segment_num = \
            compute_global_nums(pan, sem_anno_dir, category_ids_to_keep, global_segment_num)
        segment_nums = compute_stats(pan, sem_anno_dir, category_ids_to_keep)
        segment_nums = update_stats_in_memory(prev_sem_anno_dir, category_ids_to_keep_past, segment_nums, pan)

        prev_image_ids = [im[:-4] for im in os.listdir(prev_sem_anno_dir)]
        prev_images_data = {}
        for img_id in prev_image_ids:
            prev_images_data[img_id] = list(segment_nums[img_id].values())

        print("Selecting previous images...")
        selected_prev_images, current_segment_num = greedy_selection(
                images_data=prev_images_data,
                num_categories=target_segment_ratio.shape[0],
                num_selections=int(num_selections * keep_ratio),
                target_segment_ratio=target_segment_ratio
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
            current_segment_num=current_segment_num
        )
        selected_images = selected_prev_images + selected_curr_images

    else:
        target_segment_ratio, global_segment_num = \
            compute_global_nums(pan, sem_anno_dir, category_ids_to_keep)
        segment_nums = compute_stats(pan, sem_anno_dir, category_ids_to_keep)

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

    generate_sem_memory_annotations(selected_images, segment_nums, sem_anno_dir, output_sem_anno_dir)
    stats = {'global_segment_num': global_segment_num}
    with open(output_target_path, 'w') as f:
        json.dump(stats, f)


print("Preparing semantic 100-10...")
for i in range(1, 6):
    print("Step", i)
    prepare_memory_css(split='100-10', step=i)
print("Preparing semantic 100-50...")
for i in range(1, 2):
    print("Step", i)
    prepare_memory_css(split='100-50', step=i)
print("Preparing semantic 100-5...")
for i in range(1, 11):
    print("Step", i)
    prepare_memory_css(split='100-5', step=i)
print("Preparing semantic 50-50...")
for i in range(1, 3):
    print("Step", i)
    prepare_memory_css(split='50-50', step=i)
