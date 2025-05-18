# helper.py
"""
Helper functions for the project.
"""
import requests
import base64
import json
import random
import os
import shutil
from PIL import Image, ImageDraw
import time
import csv
from io import BytesIO
import copy
import numpy as np


# 从Python SDK导入BOS配置管理模块以及安全认证模块
from baidubce.services import bos
from baidubce.services.bos import canned_acl
from baidubce.services.bos.bos_client import BosClient
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials


def cal_delta(data_list):
    """
    计算列表中最大值与第一个元素的差值。

    Args:
        data_list (list): 包含数值的列表。

    Returns:
        float: 列表中最大值与第一个元素的差值。

    """
    # acc_0 = data_list[0]
    # acc_max = 0
    # for k in range(1, len(data_list)):
    #     if data_list[k] > acc_max:
    #         acc_max = data_list[k]
    # metric = (acc_max - acc_0) / (1 - acc_0)
    # # metric = 1 / (1 + np.exp(-metric))
    # return round(metric, 3)

    if (data_list.count(0) > 3):
        return 0

    acc_0 = data_list[0]
    sum_1 = 0
    sum_2 = 0
    for k in range(1, len(data_list)):
        if data_list[k] == 0:
            continue
        sum_1 += data_list[k] - acc_0
        sum_2 += 1 - acc_0
    if sum_2 == 0:
        return 0
    else:
        return round(sum_1 / sum_2, 3)


def cal_eta(data_list):
    """
    计算 ETA（预期平均时间）值。

    Args:
        data_list (list): 包含多个数据点的列表，每个数据点表示一次观测或实验的结果。

    Returns:
        float: 计算得到的 ETA 值，保留三位小数。

    描述:
        函数接收一个包含多个数据点的列表作为输入。首先，它将列表的第一个元素赋值给变量 acc_0。
        然后，它初始化两个变量 sum_1 和 sum_2，分别用于存储求和和计数。
        接下来，函数遍历列表中从第二个元素开始的所有元素。如果当前元素为 0，则跳过该元素；
        否则，将该元素与 acc_0 的差值加到 sum_1，并将 1 - acc_0 加到 sum_2。
        如果 sum_2 为 0，则返回 0；否则，返回 sum_1 / sum_2 的值，并保留三位小数。
    """
    if (data_list.count(0) > 3):
        return 0

    acc_0 = data_list[0]
    sum_1 = 0
    sum_2 = 0
    for k in range(1, len(data_list)):
        if data_list[k] == 0:
            continue
        sum_1 += data_list[k] - acc_0
        sum_2 += max(data_list) - acc_0
    if sum_2 == 0:
        return 0
    else:
        return round(sum_1 / sum_2, 3)


def cal_phi(data_list1, data_list2):
    """
    计算两个数据列表之间的phi值。

    Args:
        data_list1 (list): 第一个数据列表。
        data_list2 (list): 第二个数据列表。

    Returns:
        float: 两个数据列表之间的phi值，保留三位小数。

    Raises:
        TypeError: 如果传入的参数不是列表类型，则抛出TypeError异常。
        ValueError: 如果两个数据列表的长度不一致，则抛出ValueError异常。

    """
    sum_1 = 0
    sum_2 = 0
    for k in range(1, len(data_list1)):
        if data_list1[k] == 0 or data_list2[k] == 0:
            continue
        sum_1 += data_list1[k] - data_list2[k]
        sum_2 += data_list2[k]
    if sum_2 == 0:
        return 0
    else:
        return round(sum_1 / sum_2, 3)


def average(data_list):
    """
    计算列表的平均值

    Args:
        data_list (list): 包含数值的列表

    Returns:
        float: 列表的平均值
    """
    return sum(data_list) / len(data_list)


def process_jsonl_path(jsonl_path):
    """
    处理JSONL格式的文件，并将其保存到指定路径。

    Args:
        jsonl_path (str): JSONL文件的路径。

    Returns:
        None

    """
    shot_num = 1
    save_path = (
        "/mnt/cfs_bj/liannan/visualicl_logs/processed_data/jsonl/"
        + "manufacture-final-0424_cleaninstruct_1shots_visualintro.jsonl"
    )
    json_list = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if not "query_info" in data.keys():
                continue
            data_id = data["data_info"]["id"]
            instruct = data["query_info"]["instruct"]["clean"]
            img_path = data["query_info"]["img_path"]
            label = data["query_info"]["label"]
            demo_info = data["demo_info"]
            prompt = instruct
            img_list = []

            for i in range(0, shot_num):
                j = 0
                for label in demo_info.keys():
                    prefix = demo_info[label][i]["visual_intro"][0]
                    postfix = demo_info[label][i]["visual_intro"][1]
                    if j == len(demo_info.keys()) - 1:
                        prompt = (
                            prompt
                            + "\n"
                            + prefix
                            + " <image> "
                            + postfix
                            + "\n Please answer the category of [query_image] <image>"
                        )
                    else:
                        prompt = prompt + "\n" + prefix + " <image> " + postfix
                    j += 1
                    # prompt += "image: <image>\n output:" + demo_info[label][i]["label_text"] + "\n"
                    img_list.append(demo_info[label][i]["shot_path"])
            img_list.append(img_path)
            conversations = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": ""},
            ]
            json_list.append(
                {"id": data_id, "conversations": conversations, "images": img_list}
            )

    print(f"Processing finished with {len(json_list)} in total, saving to {save_path}")
    save_log(save_path, json_list)

    return


def convert_img_to_url(img_path, id=""):
    """
    将图片转换为URL地址。

    Args:
        img_path (str): 图片的路径。
        id (str, optional): 图片的唯一标识符。默认为空字符串。

    Returns:
        str: 图片的URL地址。

    """

    # 设置BosClient的Host，Access Key ID和Secret Access Key
    bos_host = "bj.bcebos.com"
    access_key_id = "***"
    secret_access_key = "***"

    # 创建BceClientConfiguration
    config = BceClientConfiguration(
        credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host
    )

    # 新建BosClient
    bos_client = BosClient(config)
    if len(id) < 3:
        id = "demo_" + img_path
    f = open(img_path, "rb")
    file_str = f.read()  # file_str = b"文件内容"
    object_key = "vlsualicl_evalset_manufacture_1050_final" + "_" + id
    bos_client.put_object_from_string(
        "qianfan-pm-liannan", object_key, file_str, content_type="image/jpeg"
    )
    url = bos_client.generate_pre_signed_url(
        "qianfan-pm-liannan",
        object_key,
        timestamp=int(time.time()),
        expiration_in_seconds=-1,
    )
    new_url = str(url)[2:][:-1][:4] + "s" + str(url)[2:][:-1][4:]

    return new_url


def convert_jsonl_to_csv(jsonl_file, csv_file):
    """
    将JSON Lines文件转换为CSV文件。

    Args:
        jsonl_file (str): JSON Lines文件的路径。
        csv_file (str): 输出CSV文件的路径。

    Returns:
        None

    Raises:
        None

    """
    data_list1 = []
    data_list2 = []
    data_list3 = []
    data_list4 = []
    data_list5 = []
    data_list6 = []

    id_list = []
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            print("reading...", csv_file)
            reader = csv.reader(f)
            for row in reader:
                id_list.append(row[0])
                data_list1.append(row)
    if os.path.exists(csv_file + "_1shot.csv"):
        with open(csv_file + "_1shot.csv", "r") as f:
            print("reading...", csv_file + "_1shot.csv")
            reader = csv.reader(f)
            for row in reader:
                if not row[0] in id_list:
                    print("[warning] id not found in 1shot, skip!", row[0])
                    continue
                data_list2.append(row)
    if os.path.exists(csv_file + "_3shot.csv"):
        with open(csv_file + "_3shot.csv", "r") as f:
            print("reading...", csv_file + "_3shot.csv")
            reader = csv.reader(f)
            for row in reader:
                if not row[0] in id_list:
                    print("[warning] id not found in 3shot, skip!", row[0])
                    continue
                data_list3.append(row)
    if os.path.exists(csv_file + "_5shot.csv"):
        with open(csv_file + "_5shot.csv", "r") as f:
            print("reading...", csv_file + "_5shot.csv")
            reader = csv.reader(f)
            for row in reader:
                if not row[0] in id_list:
                    print("[warning] id not found in 5shot, skip!", row[0])
                    continue
                data_list4.append(row)
    if os.path.exists(csv_file + "_1shot_fabriacteddemo.csv"):
        with open(csv_file + "_1shot_fabriacteddemo.csv", "r") as f:
            print("reading...", csv_file + "_1shot_fabriacteddemo.csv")
            reader = csv.reader(f)
            for row in reader:
                if not row[0] in id_list:
                    print(
                        "[warning] id not found in 1shot_fabriacteddemo, skip!", row[0]
                    )
                    continue
                data_list5.append(row)
    if os.path.exists(csv_file + "_1shot_randomdemo.csv"):
        with open(csv_file + "_1shot_randomdemo.csv", "r") as f:
            print("reading...", csv_file + "_1shot_randomdemo.csv")
            reader = csv.reader(f)
            for row in reader:
                if not row[0] in id_list:
                    print("[warning] id not found in 1shot_randomdemo, skip!", row[0])
                    continue
                data_list6.append(row)

    print(
        "Totally reading...",
        len(data_list1),
        len(data_list2),
        len(data_list3),
        len(data_list4),
        len(data_list5),
        len(data_list6),
    )
    print("id_list:", len(id_list), id_list)
    if len(id_list) < 1:
        data_list1 = [["id", "url", "candidate", "instruct", "demo1", "text1"]]
        data_list2 = [["id", "url", "candidate", "instruct", "demo1", "text1"]]
        data_list3 = [["id", "url", "candidate", "instruct", "demo1", "text1"]]
        data_list4 = [["id", "url", "candidate", "instruct", "demo1", "text1"]]
        data_list5 = [["id", "url", "candidate", "instruct", "demo1", "text1"]]
        data_list6 = [["id", "url", "candidate", "instruct", "demo1", "text1"]]

    i = 0
    count = 0
    count_total = 0
    with open(jsonl_file, "r") as file:
        for line in file.readlines():
            row = json.loads(line)
            if not "query_info" in row.keys():
                continue
            i += 1
            if row["data_info"]["id"] in id_list:
                print("[Skip]id found in id_list", row["data_info"]["id"])
                continue

            img_path = row["query_info"]["img_path"]
            url = convert_img_to_url(img_path, row["data_info"]["id"])
            data_list1.append(
                [
                    row["data_info"]["id"],
                    url,
                    row["query_info"]["candidate"],
                    row["query_info"]["instruct"]["clean"],
                ]
            )
            demo_info = row["demo_info"]
            temp = [
                row["data_info"]["id"],
                url,
                row["query_info"]["candidate"],
                row["query_info"]["instruct"]["clean"],
            ]
            for label in demo_info.keys():
                shot = demo_info[label][0]
                temp.append(convert_img_to_url(shot["shot_path"]))
                temp.append(shot["label_text"])
            data_list2.append(temp)

            data_list3.append(copy.deepcopy(data_list2[i]))
            data_list4.append(copy.deepcopy(data_list2[i]))
            for j in range(1, 3):
                for label in demo_info.keys():
                    shot = demo_info[label][j]
                    new_url = convert_img_to_url(shot["shot_path"])
                    data_list3[i].append(new_url)
                    data_list3[i].append(shot["label_text"])
                    data_list4[i].append(new_url)
                    data_list4[i].append(shot["label_text"])

            for j in range(3, 5):
                for label in demo_info.keys():
                    shot = demo_info[label][j]
                    data_list4[i].append(convert_img_to_url(shot["shot_path"]))
                    data_list4[i].append(shot["label_text"])

            demo_info = row["fabricated_demo_info"]
            query_info = row["fabricated_query_info"]
            data_list5.append(copy.deepcopy(data_list2[i]))
            data_list5[i][2] = query_info["candidate"]
            data_list5[i][3] = query_info["instruct"]["clean"]

            j = 0
            for label in demo_info.keys():
                data_list5[i][5 + 2 * j] = demo_info[label][0]["label_text"]
                j = j + 1

            demo_info = row["random_demo_info"]
            data_list6.append(copy.deepcopy(data_list2[i]))
            j = 0
            for label in demo_info.keys():
                # print(data_list6[i][0], row["data_info"]["id"])
                # print(demo_info[label][0]["shot_path"])
                # print(label, data_list6[i][5 + 2*j], demo_info[label][0]["label_text"])
                data_list6[i][5 + 2 * j] = demo_info[label][0]["label_text"]
                if label == data_list6[i][5 + 2 * j]:
                    count += 1
                count_total += 1
                j = j + 1

            if i % 20 == 0:
                print(f"Processing {i} data")
                with open(csv_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(data_list1)
                    print("saving...", csv_file)
                with open(csv_file + "_1shot.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(data_list2)
                    print("saving...", csv_file + "_1shot.csv")
                with open(csv_file + "_3shot.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(data_list3)
                    print("saving...", csv_file + "_3shot.csv")
                with open(csv_file + "_5shot.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(data_list4)
                    print("saving...", csv_file + "_5shot.csv")
                with open(csv_file + "_1shot_fabriacteddemo.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(data_list5)
                    print("saving...", csv_file + "_1shot_fabriacteddemo.csv")
                with open(csv_file + "_1shot_randomdemo.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(data_list6)
                    print("saving...", csv_file + "_1shot_randomdemo.csv")
                print("same random count: ", count, count_total)

    print("same random count: ", count, count_total)
    print(f"Final Saving... Processing {i} data")
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_list1)
        print("saving...", csv_file)
    with open(csv_file + "_1shot.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_list2)
        print("saving...", csv_file + "_1shot.csv")
    with open(csv_file + "_3shot.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_list3)
        print("saving...", csv_file + "_3shot.csv")
    with open(csv_file + "_5shot.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_list4)
        print("saving...", csv_file + "_5shot.csv")
    with open(csv_file + "_1shot_fabriacteddemo.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_list5)
        print("saving...", csv_file + "_1shot_fabriacteddemo.csv")
    with open(csv_file + "_1shot_randomdemo.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_list6)
        print("saving...", csv_file + "_1shot_randomdemo.csv")

    return


def average_update(data_list, new_item):
    """
    向列表中添加一个新元素，并计算更新后的列表的平均值。

    Args:
        data_list (list): 包含数值的列表。
        new_item (float): 要添加到列表中的新元素。

    Returns:
        tuple: 包含更新后的列表和新的平均值的元组。
    """
    ava = (sum(data_list) + new_item) / (len(data_list) + 1)
    data_list.append(new_item)
    return data_list, ava


def convert_bmp2png(file_path):
    """
    将BMP格式的图片文件转换为PNG格式。

    Args:
        file_path (str): BMP格式的图片文件路径。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果指定的文件路径不存在，将抛出此异常。
        IOError: 如果文件不是有效的BMP格式图片，将抛出此异常。

    """
    bmp_image = Image.open(file_path)
    bmp_image.save(file_path.split(".")[0] + ".png", "PNG")
    os.remove(file_path)


def convert_bmp2png_all(dir_path):
    """
    遍历给定目录下的所有子目录和文件，并对所有BMP格式的图片文件进行转换操作。

    Args:
        dir_path (str): 需要处理的目录路径。

    Returns:
        None

    """
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            file_path = os.path.join(root, name)
            if is_image_by_extension(file_path):
                convert_bmp2png(file_path)
                print(f"converting {file_path} to png")


def reorganize_dataset(dir_path):
    """
    重新组织数据集，将所有图片移动到一个统一的目录下。

    Args:
        dir_path (str): 数据集所在的根目录路径。

    Returns:
        None

    """
    if "BTAD" in dir_path:
        for root, dirs, files in os.walk(dir_path):
            if not "test" in root:
                continue
            for name in files:
                src_path = os.path.join(root, name)
                if is_image_by_extension(src_path):
                    object_name = name.split("_")[0]
                    label = name.split("_")[1]
                    file_name = name.split("_")[-1]
                    dest_path = os.path.join(
                        dir_path, object_name, "test", label, file_name
                    )
                    os.makedirs("/".join(dest_path.split("/")[:-1]), exist_ok=True)
                    shutil.copy(src_path, dest_path)
                    print("Copy {} to {}".format(src_path, dest_path))
    if "MVTec-3D" in dir_path:
        for root, dirs, files in os.walk(dir_path):
            for name in files:
                src_path = os.path.join(root, name)
                if not ("test" in root):
                    continue
                if not ("rgb" in root or "gt" in root):
                    continue

                if is_image_by_extension(src_path):
                    object_name = src_path.split("/")[-5]
                    index = src_path.split("/")[-2]
                    label = src_path.split("/")[-3]
                    file_name = src_path.split("/")[-1]
                    new_root = src_path.split(object_name)[0]
                    index = "test_" + index
                    dest_path = os.path.join(
                        new_root, object_name, index, label, file_name
                    )
                    dest_path = dest_path.replace("MVTec-3D", "MVTec-3D-new")
                    os.makedirs("/".join(dest_path.split("/")[:-1]), exist_ok=True)
                    shutil.copy(src_path, dest_path)
                    print("Copy {} to {}".format(src_path, dest_path))


def split_list(target_list, n):
    """
    将 target_list 平均分成 n 份，尽量做到数量接近。
    """
    k, m = divmod(len(target_list), n)
    return [
        target_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    ]


def check_success(success_path, log_path):
    """
    检查任务是否成功。

    Args:
        success_path_jsonl (str): 成功路径的 JSONL 文件名。
        log_path (str): 日志文件的路径。

    Returns:
        bool: 如果日志文件中包含表示成功的特定日志，则返回 True；否则返回 False。

    """
    try:
        with open(success_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())
                # print("Data:", data["log_path"])
                info = data["log_path"].split("logs/")[-1].split("_")[2:]
                # print("**", industry, "_".join(info))
                if "_".join(info) in log_path:
                    data_num = (
                        data["exp_result"]["Total"][1] + data["exp_result"]["error_num"]
                    )
                    return {"num": data_num, "log_path": log_path}
        return {}
    except FileNotFoundError:
        print("No such file or directory: {}".format(success_path))
        return {}


def is_image_by_extension(file_name):
    """
    判断给定的文件是否是图片类型。

    Args:
        file_name (str): 待检查的文件名称。

    Returns:
        bool: True表示是图片类型，False表示不是图片类型。

    """
    image_extensions = ["jpg", "jpeg", "png", "bmp"]
    extension = file_name.split(".")[-1].lower()
    return extension in image_extensions


def read_filename_from_dir(dir_path, file_type="img"):
    """
    从指定目录读取文件名，并返回文件名列表和目录列表。

    Args:
        dir_path (str): 目标目录的路径。

    Returns:
        tuple: 包含两个列表的元组。
            - list: 包含包含文件名（包含路径）的列表。
            - list: 包含目标目录中包含的子目录名的列表。

    """
    final_list = []
    cate_list = []
    # print("dir_path:", dir_path)
    for root, dirs, files in os.walk(dir_path):
        if len(cate_list) < 1:
            cate_list = dirs
        for file in files:
            full_path = os.path.join(root, file)
            if "MVTec-3D" in full_path:
                if not "rgb" in full_path:
                    continue

            if file_type == "img" and is_image_by_extension(file):
                final_list.append(full_path)
            if file_type == "jsonl" and "jsonl" in file:
                final_list.append(full_path)
    return final_list, cate_list


def read_filename_from_jsonl(dir_path):
    """
    从指定目录读取jsonl文件并返回两个字典和列表。

    Args:
        dir_path (str): 要搜索jsonl文件的目录路径。

    Returns:
        tuple: 包含两个元素的元组。
                - product_url_dict (dict): 包含产品URL的字典。
                - exp_data_list (list): 包含实验数据的列表。

    """
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            full_path = os.path.join(root, file)
            if "fewshot" in full_path:
                exp_path = full_path
            if "url" in full_path:
                url_path = full_path
    print(exp_path, url_path)
    with open(url_path, "r") as f:
        product_url_dict = json.load(f)
    with open(exp_path, "r") as f:
        exp_data_list = [json.loads(line) for line in f]
    print("Reading from", url_path, len(product_url_dict.keys()))
    print("Reading from", exp_path, len(exp_data_list))

    return product_url_dict, exp_data_list


def prompt_assemble(prompt_template, key, value):
    """
    将模板中的占位符替换为指定的值。

    Args:
        prompt_template (str): 模板字符串，其中包含一个或多个占位符，格式为 "{{key}}"。
        key (str): 占位符中的键，用于在模板中定位要替换的占位符。
        value (str): 用于替换占位符的值。

    Returns:
        str: 替换占位符后的新字符串。

    """
    index = "{{" + key + "}}"
    prompt = prompt_template.replace(index, value)
    return prompt


def random_sample_fewshot(img_list, img_path, target, shot_num, flag):
    """
    从图片列表中随机选择几个符合条件的图片，并生成一个字典。

    Args:
        img_list (list): 图片名称列表。
        img_path (str): 图片存储路径。
        target (str): 目标标签。
        shot_num (int): 要选择的图片数量。
        flag (bool): 选择包含目标标签的图片（True）还是不包含目标标签的图片（False）。

    Returns:
        dict: 包含选择的图片及其目标标签的字典。

    """
    condition = lambda x: target in x

    if flag:
        sample_list = [x for x in img_list if condition(x)]
    else:
        sample_list = [x for x in img_list if not condition(x)]
        selected_elements = random.sample(
            [x for x in img_list if not condition(x)], shot_num
        )
    if shot_num > len(sample_list):
        print(
            f"[Warning] {img_path} has not enough similar images with {target} label from {len(sample_list)} for {shot_num}."
        )
        print(sample_list)
        shot_num = len(sample_list)

    selected_elements = random.sample(sample_list, shot_num)

    final_list = []
    for element in selected_elements:
        label = element.split("/")[-2]
        final_list.append({"shot_path": element, "label_text": label})
    return final_list


def download_image(url, save_path, log_path):
    """
    从给定的URL下载图片并保存到指定的路径。

    Args:
        url (str): 图片的URL地址。
        save_path (str): 图片的保存路径。
        log_path (str): 日志文件的路径。

    Returns:
        int: 下载成功返回1，失败返回0。

    """
    print("# ImgDownload\n", url)
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Img saving... {save_path}")
        return 1
    else:
        print("Something wrong while saving img", url)
        with open(log_path, "a") as f:
            f.writelines(save_path + "#*#" + url + "\n")
        return 0


def save_log(log_path, json_object, mode="a"):
    """
    将 JSON 对象保存到指定的日志文件中。

    Args:
        log_path (str): 日志文件的路径。
        json_object (dict): 要保存的 JSON 对象。
        mode (str, optional): 文件打开模式，默认为 "a"。如果文件不存在，则创建新文件。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果指定的文件路径不存在，将引发 FileNotFoundError 异常。
        IOError: 如果在文件写入过程中发生错误，将引发 IOError 异常。

    """
    if mode == "f":
        updated_lines = []
        replaced = False
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if not replaced and obj.get("query_info", {}).get(
                    "img_path"
                ) == json_object.get("query_info", {}).get("img_path"):
                    updated_lines.append(json.dumps(json_object, ensure_ascii=False))
                    replaced = True
                else:
                    updated_lines.append(line.strip())
            if replaced:
                with open(log_path, "w", encoding="utf-8") as f:
                    for line in updated_lines:
                        f.write(line + "\n")
            else:
                f.write(json.dumps(json_object, ensure_ascii=False) + "\n")
    else:
        dir_path = os.path.dirname(log_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(log_path, mode, encoding="utf-8") as f_out:
            f_out.write(json.dumps(json_object, ensure_ascii=False) + "\n")


def div(a, b):
    """
    计算两个数的除法结果，并保留两位小数。

    Args:
        a (float): 被除数。
        b (float): 除数。

    Returns:
        float: 除法结果，保留两位小数。如果除数为0，则返回0。

    """
    if b == 0:
        return 0
    else:
        return round(a / b, 2)


def cal_metric(cate_result_dict):
    """
    计算评估指标。

    Args:
        cate_result_dict (dict): 包含分类结果的字典，字典的键为类别名，值为一个包含两个元素的列表，第一个元素为该类别的正确预测数，第二个元素为该类别的总预测数。

    Returns:
        dict: 包含评估指标的字典，字典的键为类别名，值为一个包含三个元素的列表，分别为该类别的正确预测数、该类别的总预测数、该类别的正确率。

    """
    total_num = 0
    total_true = 0
    json_object = {}
    print("#### Display Metric of the Eval")
    for key in cate_result_dict.keys():
        total_num += cate_result_dict[key][1]
        total_true += cate_result_dict[key][0]
        json_object[key] = [
            cate_result_dict[key][0],
            cate_result_dict[key][1],
            div(cate_result_dict[key][0], cate_result_dict[key][1]),
        ]
    print("-- Total", total_true, total_num, div(total_true, total_num))
    json_object["Total"] = [total_true, total_num, div(total_true, total_num)]
    return json_object


def download_image_from_txt(log_path):
    """
    从给定的日志文件下载图片。

    Args:
        log_path (str): 日志文件的路径。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果日志文件不存在，将引发此异常。
        ValueError: 如果日志文件的行格式不正确，将引发此异常。

    """
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            save_path = line.split("#*#")[0]
            url = line.split("#*#")[1]
            print("path:", save_path, "url:", url)
            download_image(url, save_path, log_path)


def found_mask_path(img_path):
    """
    根据图像路径找到对应的掩膜路径。

    Args:
        img_path (str): 图像路径。

    Returns:
        str: 如果找到对应的掩膜路径，则返回该路径；否则返回 -1。

    """
    # print(img_path)
    mask_path = ""
    if "/image/" in img_path:
        mask_path = img_path.replace("/image/", "/mask/")
    elif "/test" in img_path:
        if "MVTec-AD/" in img_path:
            mask_path = img_path.replace("/test/", "/ground_truth/")
            mask_path = mask_path.split(".")[0] + "_mask.png"
        if "MVTec-LOCO" in img_path:
            mask_path = img_path.replace("/test/", "/ground_truth/")
            filename = mask_path.split("/")[-1]
            prefix = filename.split(".")[0]
            mask_path = (
                "/".join(mask_path.split("/")[:-1]) + "/" + prefix + "/" + "000.png"
            )
        if "MVTec-AD-2" in img_path:
            mask_path = img_path.replace("/test_public/", "/ground_truth/")
            mask_path = mask_path.split(".")[0] + "_mask.png"
        if "MVTec-3D-new" in img_path:
            mask_path = img_path.replace("/test_rgb/", "/test_gt/")
            mask_path = mask_path.split(".")[0] + ".png"
        if "ITD" in img_path:
            mask_path = img_path.replace("/test/", "/ground_truth/")
            mask_path = mask_path.split(".")[0] + "_mask.png"
        if "ISP-AD" in img_path:
            mask_path = img_path.replace("/test/", "/ground_truth/")
            mask_path = mask_path.split(".")[0] + ".png"
    else:
        return ""
    mask_path = mask_path.split(".")[0] + ".png"
    if not os.path.isfile(mask_path):
        return ""
    return mask_path


def encode_image_withmask(img_path, mask_demo_flag):
    mask_path = found_mask_path(img_path)
    if len(mask_path) < 3:
        return encode_image(img_path)

    if mask_demo_flag == -1:
        inset_margin = 10
        padding = 10
        img = Image.open(img_path).convert("RGB")
        # out_path = "./src/images/test_original.png"
        # img.save(out_path, format="PNG")
        mask = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)

        bbox = mask.getbbox()  # (left, upper, right, lower)
        # print("**", bbox)
        if bbox is None:
            return encode_image(img_path)
        l, u, r, d = bbox
        l = max(l - padding, 0)
        u = max(u - padding, 0)
        r = min(r + padding, img.width)
        d = min(d + padding, img.height)
        new_bbox = (l, u, r, d)

        detail_crop = img.crop(new_bbox)

        # 3. 放大细节
        w, h = detail_crop.size
        zoom = min(0.4 * img.size[0] / h, 0.4 * img.size[1] / w)
        zoomed = detail_crop.resize((int(w * zoom), int(h * zoom)), Image.BICUBIC)

        # 4. 在原图左上角贴入放大图（可留边距）
        img.paste(zoomed, (inset_margin, inset_margin))

        frame_width = 4
        frame_color = (255, 0, 0)
        draw = ImageDraw.Draw(img)
        for i in range(frame_width):  # 多次描边以获得指定宽度
            draw.rectangle(
                [l - i, u - i, r + i - 1, d + i - 1],
                outline=frame_color,
            )

        # 5. 转 Base64
        # out_path = "./src/images/test_zoom.png"
        # img.save(out_path, format="PNG")
        # print("saving to", out_path)
        # return True

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    else:
        return encode_image(img_path)


def encode_image(image_path):
    """
    将图片文件编码为Base64格式。

    Args:
        image_path (str): 图片文件的路径。

    Returns:
        str: 图片文件的Base64编码字符串。

    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        if (
            "jp" in image_path.split("/")[-1].split(".")[-1]
            or "JP" in image_path.split("/")[-1].split(".")[-1]
        ):
            url = f"data:image/jpeg;base64,{base64_image}"
        elif (
            "pn" in image_path.split("/")[-1].split(".")[-1]
            or "PN" in image_path.split("/")[-1].split(".")[-1]
        ):
            url = f"data:image/png;base64,{base64_image}"
        elif (
            "bm" in image_path.split("/")[-1].split(".")[-1]
            or "BM" in image_path.split("/")[-1].split(".")[-1]
        ):
            url = f"data:image/bmp;base64,{base64_image}"
        else:
            print("#Checking... \n Image Type is not found!", image_path)
            return

    return url
