# img_process.py
"""
This module will visualize the results of the jsonl file
"""

import argparse
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import numpy as np
import os
from utils.helper import (
    read_filename_from_dir,
    random_sample_fewshot,
    reorganize_dataset,
    convert_jsonl_to_csv,
    convert_img_to_url,
    found_mask_path,
)
import csv
import time



def generate_maskedimg_from_path(
    img_path, mask_path, save_path=None, box_size=(100, 100), sparse_level=1
):
    """
    从指定路径生成带有掩码的图像。

    Args:
        img_path (str): 输入图像的路径。
        mask_path (str): 掩码图像的路径。
        save_path (str, optional): 保存结果图像的路径。默认为None，表示不保存图像。
        box_size (tuple, optional): 矩形框的大小（宽，高）。默认为(100, 100)。
        sparse_level (int, optional): 稀疏程度，表示掩码区域内需要修改的像素点比例。默认为1。

    Returns:
        bool: 如果成功生成并保存图像，则返回True；否则返回False。

    Raises:
        ValueError: 如果原始掩码中没有正像素点，则抛出此异常。
    """

    # 打开图像
    if not os.path.isfile(img_path):
        print(f"image path not exist {img_path}")
        return False
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    if os.path.isfile(mask_path):
        
        mask = Image.open(mask_path).convert("L")  # 灰度模式
        # 转换为 numpy 数组
        mask_np = np.array(mask)
        # 找到原mask中非零像素的中心
        ys, xs = np.where(mask_np > 0)

        if len(xs) == 0 or len(ys) == 0:
            center_x = random.randint(0, img_np.shape[1] - 1)
            center_y = random.randint(0, img_np.shape[0] - 1)
        else:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
        # print("---***---")
        # print(mask_np.shape)
        # print(center_x, center_y)
    else:
        center_x = random.randint(0, img_np.shape[1] - 1)
        center_y = random.randint(0, img_np.shape[0] - 1)


    # 矩形box的大小
    box_w, box_h = box_size

    # 计算矩形左上角和右下角坐标
    x1 = max(center_x - box_w // 2, 0)
    y1 = max(center_y - box_h // 2, 0)
    x2 = min(center_x + box_w // 2, img_np.shape[1] - 1)
    y2 = min(center_y + box_h // 2, img_np.shape[0] - 1)
    # print(x1,y1,x2,y2)

    # 创建新的空白mask

    new_mask = Image.new("L", img_np.shape[0:2], 0)  # 全黑背景
    draw = ImageDraw.Draw(new_mask)
    before = new_mask.getpixel((center_y, center_x))
    draw.rectangle([y1, x1, y2, x2], fill=255)
    after = new_mask.getpixel((center_y, center_x))

    # print(before, after)

    new_mask_np = np.array(new_mask)
    # print(np.sum(new_mask_np))
    # 找出mask区域的所有像素位置（非0）
    mask_indices = np.argwhere(new_mask_np > 0)
    total_mask_pixels = len(mask_indices)

    if total_mask_pixels == 0:
        print("Mask中没有可处理区域", img_path)
        return False

    # 随机选择一半的位置进行替换
    num_to_modify = total_mask_pixels // sparse_level
    selected_indices = mask_indices[
        np.random.choice(total_mask_pixels, num_to_modify, replace=False)
    ]

    # 应用随机噪声
    for y, x in selected_indices:
        img_np[x, y] = [random.randint(0, 255) for _ in range(3)]

    # 转换回PIL图像
    result_img = Image.fromarray(img_np)

    # 保存图像（如果需要）
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        result_img.save(save_path)
        #print(f"saving to {save_path}")
    return True


def cal_box_size_from_shot(shot_info):
    """
    根据镜头信息计算盒子尺寸。

    Args:
        shot_info (dict): 镜头信息字典，包含多个镜头的信息。

    Returns:
        tuple: 返回两个整数的元组，表示盒子的长度和宽度。如果只有一个镜头，则返回(-1, -1)。

    Raises:
        ValueError: 如果镜头蒙版中没有正像素，则引发此异常。

    """
    size_list = [[], []]
    #print(shot_info.keys())
    for key in shot_info:
        mask_path = found_mask_path(shot_info[key][0]["shot_path"])
        if not os.path.isfile(mask_path):
            #print(f"not exist! {mask_path}")
            continue
        #else:
            #print(f"found! {mask_path}")
        mask = Image.open(mask_path).convert("L")  # 灰度模式
        # 转换为 numpy 数组
        mask_np = np.array(mask)
        # 找到原mask中非零像素的中心
        ys, xs = np.where(mask_np > 0)
        # print(ys, xs)

        if len(xs) == 0 or len(ys) == 0:
            continue

        length = int(np.max(xs)) - int(np.min(xs))
        width = int(np.max(ys)) - int(np.min(ys))
        size_list[0].append(length)
        size_list[1].append(width)
    #print(size_list)
    if len(size_list[0]) > 1:
        return (int(np.median(size_list[0])), int(np.median(size_list[1])))
    else:
        return (-1, -1)



def generate_maskedimg_from_single_path(img_path, sparse_level=1, mode="each"):
    """
    从指定目录中生成掩码图像。

    Args:
        dir_path (str): 指定图像的目录路径。
        sparse_level (int, optional): 掩码图像的稀疏级别。默认为1。
        mode (str, optional): 生成掩码图像的模式。可以是"ava"或"each"。默认为"each"。

    Returns:
        None

    Raises:
        ValueError: 如果掩码图像没有正像素，则引发此异常。

    """
    dir_path = "/".join(img_path.split("/")[:-2])
    print(dir_path)
    img_list, cata_list = read_filename_from_dir(dir_path, "img")
    # print(img_list)
    # print(f"Processing {img_path}")
    mask_path = found_mask_path(img_path)
    if mask_path == -1:
        return -1
    if mode == "ava":
        demo_info = {}
        for label in cata_list:
            demo_info[label] = random_sample_fewshot(img_list, img_path, label, 1, True)
        box_size = cal_box_size_from_shot(demo_info)
    else:
        mask = Image.open(mask_path).convert("L")  # 灰度模式
        mask_np = np.array(mask)
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Shot mask has no positive pixels!")
        length = int(np.max(xs)) - int(np.min(xs))
        width = int(np.max(ys)) - int(np.min(ys))
        box_size = (length, width)

    if box_size[0] < 0:
        return -1
    save_path = "./src/images/test_image.png"
    # save_path = (
    #    img_path.replace(
    #        "Manufacture", "Manufacture-mask-s" + str(sparse_level)
    #    ).split(".")[0]
    #    + ".png"
    # )
    generate_maskedimg_from_path(img_path, mask_path, save_path, box_size, sparse_level)

    print(f"Processing image... {save_path}")
    return save_path


def generate_maskedimg_from_dir(dir_path, sparse_level=1, mode="each"):
    """
    从指定目录中生成掩码图像。

    Args:
        dir_path (str): 指定图像的目录路径。
        sparse_level (int, optional): 掩码图像的稀疏级别。默认为1。
        mode (str, optional): 生成掩码图像的模式。可以是"ava"或"each"。默认为"each"。

    Returns:
        None

    Raises:
        ValueError: 如果掩码图像没有正像素，则引发此异常。

    """
    img_list, cata_list = read_filename_from_dir(dir_path, "img")
    # print(img_list)
    i = 0
    succeed = 0

    for img_path in img_list:
        i += 1
        if i % 30 == 0:
            print(f"Processing {i} image and saving {succeed} with {len(img_list)} in total... {img_path}")
        # print(f"Processing {img_path}")
        mask_path = found_mask_path(img_path)
        save_path = (
            img_path.replace(
                "Manufacture", "Manufacture-mask-s" + str(sparse_level)
            ).split(".")[0]
            + ".png"
        )
        if(os.path.exists(save_path)):
            #pirnt("exist and skip")
            continue

        if mask_path == -1:
            #print(f"[Warning] no mask path found! random! {img_path}")
            demo_info = {}
            for label in cata_list:
                demo_info[label] = random_sample_fewshot(
                    img_list, img_path, label, 1, True
                )
            box_size = cal_box_size_from_shot(demo_info)
        else:
            mask = Image.open(mask_path).convert("L")  # 灰度模式
            mask_np = np.array(mask)
            ys, xs = np.where(mask_np > 0)
            if len(xs) == 0 or len(ys) == 0:
                #print(f"[Warning] shot mask has no positive pixels! random! {img_path}")
                demo_info = {}
                for label in cata_list:
                    demo_info[label] = random_sample_fewshot(
                        img_list, img_path, label, 1, True
                    )
                box_size = cal_box_size_from_shot(demo_info)
            else:
                length = int(np.max(xs)) - int(np.min(xs))
                width = int(np.max(ys)) - int(np.min(ys))
                box_size = (length, width)
        if box_size[0] < 0:
            demo_info = {}
            for label in cata_list:
                demo_info[label] = random_sample_fewshot(
                    img_list, img_path, label, 1, True
                )
            box_size = cal_box_size_from_shot(demo_info)
        # save_path = "./src/images/test_image.png"
        if generate_maskedimg_from_path(img_path, mask_path, save_path, box_size, sparse_level):
            succeed += 1


def generate_maskedimg_from_jsonl(jsonl_path):
    """
    从JSONL文件中生成带掩码的图像。

    Args:
        jsonl_path (str): JSONL文件的路径。

    Returns:
        None

    """
    with open(jsonl_path, "r") as f:
        data_list = [json.loads(line) for line in f]

    for data in data_list:
        if not "demo_info" in data.keys():
            continue
        img_path = data["query_info"]["img_path"]
        mask_path = found_mask_path(img_path)
        print(f"Processing {img_path}")
        shot_info = data["demo_info"]
        box_size = cal_box_size_from_shot(shot_info)
        if box_size[0] < 0:
            continue
        save_path = img_path.replace("Manufacture", "Manufacture-mask")
        generate_maskedimg_from_path(img_path, mask_path, save_path, box_size)


def generate_random_img():
    """
    生成并保存随机噪声图像和一张白色图像。

    Args:
        无参数。

    Returns:
        无返回值。

    Raises:
        无异常抛出。

    """
    output_dir = "src/images"
    os.makedirs(output_dir, exist_ok=True)

    # 图像尺寸
    width, height = 256, 256

    # 生成并保存 10 张噪声图像
    for i in range(10):
        # 生成随机噪声数据
        noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # 创建图像对象
        noise_image = Image.fromarray(noise_array, "RGB")

        # 保存图像
        filename = f"noise_image_{i+1}.png"
        noise_image.save(os.path.join(output_dir, filename))

    white_image = Image.new("RGB", (width, height), color="white")
    filename = "white_image.png"
    white_image.save(os.path.join(output_dir, filename))


if __name__ == "__main__":

    # jsonl_file = "/mnt/cfs_bj/liannan/visualicl_logs/processed_data/stage2/dataset/2025-04-24/Manufacture_manufacture-final_0.21_4_1050_11:54:51.476301_each_5_shots_demotextflag2_finepromptflag2_randomedemoflag4_maskdemoflag1.jsonl"
    # save_file = "./src/images/Manufacture_manufacture-final_0.21_4_1050_11:54:51.476301_human.csv"
    # convert_jsonl_to_csv(jsonl_file, save_file)

    # generate_maskedimg_from_single_path(
    #    "/mnt/cfs_bj/liannan/visualicl_raw_data/Manufacture/"
    #    + "MVTec-LOCO/splicing_connectors/test/structural_anomalies/084.png",
    #    3,
    # )
    # generate_maskedimg_from_single_path(
    #    "/mnt/cfs_bj/liannan/visualicl_raw_data/Manufacture/MVTec-AD/cable/test/bent_wire/000.png",
    #    3,
    # )

    parser = argparse.ArgumentParser(description="eval the dataset")
    parser.add_argument("--benchmark_dir", required=True, help="benchmark_dir")
    parser.add_argument("--sparse_level", required=True, help="sparse_level")
    args = parser.parse_args()
    generate_maskedimg_from_dir(args.benchmark_dir, int(args.sparse_level))
