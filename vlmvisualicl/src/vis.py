# vis.py
"""
This module will visualize the results of the jsonl file
"""
import json
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np
import os
from utils.helper import (
    reorganize_dataset,
    convert_bmp2png_all,
    convert_jsonl_to_csv,
    prompt_assemble,
    encode_image,
    encode_image_withmask,
    process_jsonl_path,
    average,
    cal_eta,
    cal_phi,
    cal_delta,
)
from utils.api_call import vlm_model_api_gate
from matplotlib.ticker import MaxNLocator


def reasoning_exp_process(jsonl_file):
    """
    对给定的jsonl文件进行处理，提取图像分类任务中的关键信息，并生成推理说明和问题列表。

    Args:
        jsonl_file (str): 包含图像分类任务数据的jsonl文件路径。

    Returns:
        List[str]: 包含所有问题答案的列表。

    """
    data_dict = {}
    answer_dict = {}
    model = "doubao-1.5-vision-pro-32k-250115"
    instruct_template = "\n## Task\n You should classify images into categories of {{candidate}}. This is a image of {{label}} You should think about why it is classifed as {{label}} and what is the key feature of this class. Then at least ask three individual yes/no question to determiner whether another image could be recognized as {{label}} with very detail obsearving more than 100 wrods \n If all of answer to these question is true, the image should be classified as {{label}}\n## Format\n <question>xxxx<\question>"
    with open(jsonl_file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            if not "query_info" in data.keys():
                continue
            if not data["data_info"]["source"] == "MVTec-AD":
                continue
            img_path = data["query_info"]["img_path"]
            label = data["query_info"]["label"]
            basic_prompt = data["query_info"]["instruct"]["clean"].split("##")[1]
            candidate = data["query_info"]["candidate"]
            demo_info = data["demo_info"]
            data_dict[img_path] = {
                "label": label,
                "basic_prompt": basic_prompt,
                "candidate": candidate,
            }

            posi_url = encode_image(demo_info[label][0]["shot_path"])
            instruct_template = data_dict[img_path]["basic_prompt"] + instruct_template
            prompt = prompt_assemble(instruct_template, "label", label)
            prompt = prompt_assemble(prompt, "candidate", str(candidate))
            result = vlm_model_api_gate(prompt, [], "", model)
            print(label, img_path)
            # print(result)
            question_list = result.split("<question>")
            url = encode_image(img_path)
            answer_list = []
            for question in question_list[1:]:
                question = question.split("</question>")[0].split("<\question>")[0]
                prompt = (
                    basic_prompt
                    + "\n"
                    + question
                    + "Please give detail observation of the image in Chinese related to "
                    + "the question and give your answer in the formatting: <answer> yes/no </answer>"
                )
                answer = vlm_model_api_gate(prompt, [], url, model)
                print(question)
                print(answer)
                answer = answer.split("<answer>")[1].split("</answer>")[0]
                if answer in answer_dict.keys():
                    answer_dict[answer] += 1
                else:
                    answer_dict[answer] = 1
                answer_list.append(answer)

            print("---***---", answer_dict)

    return answer_list


def cal_gap_list(gap_list, level_id=3):
    """
    计算图像路径列表的差距列表。

    Args:
        gap_list (list): 图像路径列表，列表中的每个元素是一个包含图像路径的列表。
        level_id (int, optional): 等级标识符，用于指定如何构建索引。默认值为3。

    Returns:
        dict: 一个字典，键是构建的索引，值是包含两个元素的列表，第一个元素是图像路径列表的索引，第二个元素是计数。

    """
    result_dict = {}
    for i in range(0, len(gap_list)):
        item_list = gap_list[i]
        for img_path in item_list:
            img_path = img_path.split("raw_data/")[-1]
            if level_id == 3:
                index = "*".join(
                    [
                        img_path.split("/")[1],
                        img_path.split("/")[2],
                        img_path.split("/")[3],
                    ]
                )
            elif level_id == 2:
                index = "*".join([img_path.split("/")[1], img_path.split("/")[2]])
            elif level_id == 1:
                index = "*".join([img_path.split("/")[1]])
            if index in result_dict.keys():
                result_dict[index][i] += 1
            else:
                result_dict[index] = [0, 0]
                result_dict[index][i] += 1

    return result_dict


def comparsion_among_exps(succeess_exp_jsonl, parrallel_exp_jsonl):
    """
    根据成功实验的jsonl文件，对实验进行对比分析。

    Args:
        succeess_exp_jsonl (str): 成功实验的jsonl文件路径。

    Returns:
        None

    Raises:
        ValueError: 如果输入的文件路径无效或文件内容格式错误。

    """
    exp_dict = {}
    index_list = []
    with open(succeess_exp_jsonl, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            log_path = data["log_path"]
            model = data["model"]
            shot_num = data["shot_num"]
            refine_prompt_flag = data["refine_prompt_flag"]
            demo_text_flag = data["demo_text_flag"]
            random_demo_flag = data["random_demo_flag"]
            mask_demo_flag = data["mask_demo_flag"]
            index = (
                "shotnum"
                + str(shot_num)
                + "_refineprompt"
                + str(refine_prompt_flag)
                + "_demotext"
                + str(demo_text_flag)
                + "_randomdemo"
                + str(random_demo_flag)
                + "_maskdemo"
                + str(mask_demo_flag)
                + "_expid"
                + str(0)
            )

            # print(shot_num, refine_prompt_flag, demo_text_flag, random_demo_flag)
            if model in exp_dict.keys():
                exp_dict[model][index] = log_path
            else:
                exp_dict[model] = {index: log_path}
            if not index in index_list:
                index_list.append(index)

    with open(parrallel_exp_jsonl, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            log_path = data["log_path"]
            model = data["model"]
            shot_num = data["shot_num"]
            refine_prompt_flag = data["refine_prompt_flag"]
            demo_text_flag = data["demo_text_flag"]
            random_demo_flag = data["random_demo_flag"]
            mask_demo_flag = data["mask_demo_flag"]
            index = (
                "shotnum"
                + str(shot_num)
                + "_refineprompt"
                + str(refine_prompt_flag)
                + "_demotext"
                + str(demo_text_flag)
                + "_randomdemo"
                + str(random_demo_flag)
                + "_maskdemo"
                + str(mask_demo_flag)
                + "_expid"
                + str(1)
            )

            # print(shot_num, refine_prompt_flag, demo_text_flag, random_demo_flag)
            if model in exp_dict.keys():
                exp_dict[model][index] = log_path
            else:
                exp_dict[model] = {index: log_path}

    # print(exp_dict)
    # print(pick_exp_result(exp_dict, "gpt-4o", 1, 0, 0, 0))

    ## consistency analysis
    model_target = [
        "random-policy",
        "doubao-1.5-vision-pro-32k-250115",
        "gpt-4o",
        "gemini-2.5-pro-exp-03-25",
    ]
    blank_influence = {}
    for model in model_target:
        comparsion_list = [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 1],
        ]
        jsonl_file_dict = {}

        if len(comparsion_list) < 1:
            print(model, len(exp_dict[model].keys()), "exps!")
            for key in exp_dict[model].keys():
                shot_num = int(key.split("shotnum")[-1][0])
                refine_prompt_flag = int(key.split("refineprompt")[-1][0])
                demo_text_flag = int(key.split("demotext")[-1][0])
                random_demo_flag = int(key.split("randomdemo")[-1][0])
                mask_demo_flag = int(key.split("maskdemo")[-1][0])
                exp_id = int(key.split("expid")[-1][0])

                jsonl_file_dict[key] = pick_exp_result(
                    exp_dict,
                    model,
                    shot_num,
                    refine_prompt_flag,
                    demo_text_flag,
                    random_demo_flag,
                    mask_demo_flag,
                    exp_id,
                )
        else:
            print(model, len(comparsion_list), "exps!")
            for key in comparsion_list:
                shot_num = key[0]
                refine_prompt_flag = key[1]
                demo_text_flag = key[2]
                random_demo_flag = key[3]
                mask_demo_flag = key[4]
                exp_id = key[5]
                index = "_".join([str(item) for item in key])
                # print("**", index)
                result = pick_exp_result(
                    exp_dict,
                    model,
                    shot_num,
                    refine_prompt_flag,
                    demo_text_flag,
                    random_demo_flag,
                    mask_demo_flag,
                    exp_id,
                )

                if type(result) == type("abc"):
                    jsonl_file_dict[index] = result

        # jsonl_file_dict["random-policy"] = pick_exp_result(exp_dict, "random-policy", 1, 0, 0, 1)
        # jsonl_file_dict = {
        #     "right_demo": pick_exp_result(exp_dict, model, 1, 0, 0, 1),
        #     "exchange_demo": pick_exp_result(exp_dict, model, 1, 0, 0, 2)
        # }
        # jsonl_file_dict = {"test_check": "xxx", "original:": "xxx"}

        cal_dict = {}
        repeat_dict = {}
        # print(jsonl_file_dict.keys())
        for key in jsonl_file_dict.keys():
            jsonl_file = jsonl_file_dict[key]
            # print("Processing", key)
            with open(jsonl_file, "r") as f:
                for line in f.readlines():
                    # print(jsonl_file, line)
                    data = json.loads(line)
                    if not "img_path" in data.keys():
                        continue
                    img_path = data["img_path"]
                    result = data["mark"]
                    # result = data["result"]
                    if key in cal_dict.keys():
                        if img_path in cal_dict[key].keys():
                            if img_path in repeat_dict.keys():
                                repeat_dict[img_path] += 1
                            else:
                                repeat_dict[img_path] = 1
                            continue
                        else:
                            cal_dict[key][img_path] = result
                    else:
                        cal_dict[key] = {}
                        cal_dict[key][img_path] = result

        print("repeat num: ", len(repeat_dict))
        # print(cal_dict)

        num = len(cal_dict.keys())
        same_matric = np.zeros((num, num))
        error_matric = np.zeros((num, num))
        exp_list = list(cal_dict.keys())
        print("exp_list: ", exp_list)
        for i in range(0, num):
            for j in range(0, num):

                repeat_num = 0
                same_num = 0
                good_num = 0
                error_num = 0
                vary_num = 0
                gap_num = [0, 0]
                gap_list = [[], []]
                for img_path in cal_dict[exp_list[i]]:
                    if not (
                        img_path in cal_dict[exp_list[i]]
                        and img_path in cal_dict[exp_list[j]]
                    ):
                        vary_num += 1
                        continue
                    if not type(cal_dict[exp_list[i]][img_path]) == type(True):
                        error_num += 1
                        continue
                    if not type(cal_dict[exp_list[j]][img_path]) == type(True):
                        error_num += 1
                        continue

                    if (
                        cal_dict[exp_list[i]][img_path]
                        == cal_dict[exp_list[j]][img_path]
                    ):
                        same_num += 1
                    else:
                        if cal_dict[exp_list[i]][img_path]:
                            gap_num[0] += 1
                            gap_list[0].append(img_path)
                        if cal_dict[exp_list[j]][img_path]:
                            gap_num[1] += 1
                            gap_list[1].append(img_path)
                    good_num += 1
                same_matric[i][j] = round(same_num / good_num, 2)
                error_matric[i][j] = round(error_num / (good_num + error_num), 2)
                if i == 1 and j == 4:
                    blank_influence[model] = [
                        cal_gap_list(gap_list, 1),
                        cal_gap_list(gap_list, 2),
                    ]
                # if(i == 0 or i == 1):
                # print(f"-{i}-{j}-{vary_num}")
                # print("## gap anlaysis (dataset-level):\n", cal_gap_list(gap_list, 1))
                # print("## gap anlaysis (object-level):\n", cal_gap_list(gap_list, 2))
                # print("## gap anlaysis (class-level):\n", cal_gap_list(gap_list, 3))
                # print(same_num, good_num, [len(item) for item in gap_list])

        print("same_matric:\n", same_matric)
        print("error_matric:\n", error_matric)

        print("----****----")

    ## up and down anaylsis
    up_dict = {}
    down_dict = {}
    level_id = 0
    for model in blank_influence.keys():
        print(model)
        for key in blank_influence[model][level_id].keys():
            if not key in up_dict.keys():
                up_dict[key] = []
            if not key in down_dict.keys():
                down_dict[key] = []
            if (
                blank_influence[model][level_id][key][1]
                / (blank_influence[model][level_id][key][0] + 0.01)
                > 1.3
            ):
                up_dict[key].append([model, blank_influence[model][level_id][key]])
            if (
                blank_influence[model][level_id][key][1]
                / (blank_influence[model][level_id][key][0] + 0.01)
                < 0.7
            ):
                down_dict[key].append([model, blank_influence[model][level_id][key]])

    print("---UP---")
    for key in up_dict.keys():
        if len(up_dict[key]) > 0:
            print(key, up_dict[key])
    print("---DOWN---")
    for key in down_dict.keys():
        if len(down_dict[key]) > 0:
            print(key, down_dict[key])


def display_model_result(exp_dict_list, dataset_list):
    """
    显示模型结果

    Args:
        exp_dict_list (list): 包含多个实验结果的字典列表

    Returns:
        None

    """
    # main experiment
    print("---*** Main Exp ***---")
    model_dict = {}
    for i in range(0, len(exp_dict_list)):
        exp_dict = exp_dict_list[i]
        for model in exp_dict.keys():
            count = 0
            for j in range(0, 6):
                result = pick_exp_result(exp_dict, model, j, 0, 0, 0, 0)
                if result["Total"][-1] > 0:
                    count += 1
            if model in model_dict.keys():
                model_dict[model][dataset_list[i]] = count
            else:
                model_dict[model] = {dataset_list[i]: count}

    for model in model_dict.keys():
        print(f" ** {model} **")
        for dataset in model_dict[model].keys():
            print(f" --- {dataset} with {model_dict[model][dataset]}")

    print("--------")

    return


def pick_exp_result(
    exp_dict,
    model_name,
    shot_num,
    refine_prompt_flag=0,
    demo_text_flag=0,
    random_demo_flag=0,
    mask_demo_flag=0,
    exp_id=0,
):
    """
    从实验字典中获取特定条件下的实验结果。

    Args:
        exp_dict (dict): 实验结果的字典，字典的键为模型名称，值为包含不同实验条件下结果的字典。
        model_name (str): 模型名称，应与exp_dict中的键匹配。
        shot_num (int): 实验中的示例数量。
        refine_prompt_flag (int, optional): 是否对提示进行微调，默认为0（不进行微调）。
        demo_text_flag (int, optional): 是否包含演示文本，默认为0（不包含）。
        random_demo_flag (int, optional): 是否随机选择演示文本，默认为0（不随机选择）。

    Returns:
        dict: 包含实验结果的字典，其中包含三个元素的列表，分别代表不同的指标。如果未找到对应条件的结果，则返回{"Total": [0, 0, 0]}。

    """
    index = (
        "shotnum"
        + str(shot_num)
        + "_refineprompt"
        + str(refine_prompt_flag)
        + "_demotext"
        + str(demo_text_flag)
        + "_randomdemo"
        + str(random_demo_flag)
        + "_maskdemo"
        + str(mask_demo_flag)
        + "_expid"
        + str(exp_id)
    )
    if model_name in exp_dict.keys():
        if index in exp_dict[model_name].keys():
            return exp_dict[model_name][index]
    return {"error_num": 0, "Total": [0, 0, 0]}


def combine_model_result(exp_dict_list):
    """
    将多个实验结果的字典合并为一个结果字典，并返回合并后的结果字典和每个模型有效索引的统计信息。

    Args:
        exp_dict_list (list of dict): 包含多个实验结果的字典列表，每个实验结果字典的键为模型名称，值为另一个字典，该字典的键为实验索引，值为包含错误数和总数量的字典。

    Returns:
        tuple: 包含两个元素的元组。
            - exp_dict (dict): 合并后的结果字典，键为模型名称，值为另一个字典，该字典的键为实验索引，值为包含错误数和总数量的字典。
            - index_co (dict): 每个模型有效索引的统计信息字典，键为模型名称，值为另一个字典，该字典的键为实验索引，值为该索引出现的次数。

    """
    model_list = [
        "random-policy",
        "human",
        "doubao-1.5-vision-pro-32k-250115",
        "Qwen*Qwen2.5-VL-72B-Instruct",
        "deepseek-ai*deepseek-vl2",
        "claude-3-7-sonnet-20250219",
        "gpt-4.1",
        "gpt-4o",
        "gemini-2.5-pro-exp-03-25",
        "<gpt>o3",
        "claude-3-7-sonnet-20250219-thinking",
        "Qwen*QVQ-72B-Preview",
    ]
    index_required = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 0, -1], [2, 0, 0, 0]]
    index_list = []
    for item in index_required:
        for shot_num in range(0, 6):
            index = (
                        "shotnum"
                        + str(shot_num)
                        + "_refineprompt"
                        + str(item[0])
                        + "_demotext"
                        + str(item[1])
                        + "_randomdemo"
                        + str(item[2])
                        + "_maskdemo"
                        + str(item[3])
                        + "_expid"
                        + str(0)
                    )
            index_list.append(index)


    exp_dict = {}
    index_co = {}
    for model in model_list:
        index_co[model] = {}
        exp_dict[model] = {}
        for i in range(0, 3):
            if not model in exp_dict_list[i].keys():
                print("model not found")
                continue
            for index in exp_dict_list[i][model].keys():
                if ( not index in index_list ):
                    continue
                if (
                    exp_dict_list[i][model][index]["error_num"]
                    > exp_dict_list[i][model][index]["Total"][1]
                ):
                    continue
                if not index in exp_dict_list[i][model].keys():
                    print("too many errors")
                    continue

                if index in exp_dict[model].keys():
                    error_num = (
                        exp_dict[model][index]["error_num"]
                        + exp_dict_list[i][model][index]["error_num"]
                    )
                    Total = [
                        exp_dict[model][index]["Total"][0]
                        + exp_dict_list[i][model][index]["Total"][0],
                        exp_dict[model][index]["Total"][1]
                        + exp_dict_list[i][model][index]["Total"][1],
                        0,
                    ]
                    exp_dict[model][index] = {"error_num": error_num, "Total": Total}
                else:
                    exp_dict[model][index] = {
                        "error_num": exp_dict_list[i][model][index]["error_num"],
                        "Total": exp_dict_list[i][model][index]["Total"],
                    }

                if index in index_co[model]:
                    index_co[model][index].append(i)
                else:
                    index_co[model][index] = [i]

    miss_exp = [{}, {}, {}, {}]
    for model in index_co.keys():
        for index in index_co[model].keys():
            if len(index_co[model][index]) < 3:
                result = exp_dict[model][index]
                exp_id = index_co[model][index]
                #print(f" pop {model} with {index} due to {result} with exp {exp_id}")
                exp_dict[model].pop(index)
                for item in range(0, 4):
                    if not item in exp_id:
                        if (model in miss_exp[item].keys()):
                            miss_exp[item][model].append(index)
                        else:
                            miss_exp[item][model] = [index]
            else:
                result = exp_dict[model][index]
                #print(f" maintain {model} with {index} due to {result}")
                exp_dict[model][index]["Total"][-1] = round(
                    exp_dict[model][index]["Total"][0]
                    / exp_dict[model][index]["Total"][1],
                    2,
                )
        if len(exp_dict[model].keys()) < 1:
            print(f" pop {model} due to no valid index")
            exp_dict.pop(model)
        if len(index_co[model].keys()) < 1:
            print(f" pop {model} due to no valid index")
            index_co.pop(model)

    print("\n")
    for i in range(0, len(miss_exp)):
        print (f"\n** industry {i} missing the following exps")
        for model in miss_exp[i]:
            exps = miss_exp[i][model]
            print(f"-- {model} with {len(exps)}") #: {exps}")


    return exp_dict, index_co

def cal_eta_delta_from_exp_result(exp_dict, target_dict, filter_threshold):

    num_shot = 5
    x = range(0, num_shot + 1)  # x轴坐标
    # exp_dict = exp_dict_list[3]
    model_list = list(target_dict.keys())
    result_dict = {}
    result = ""
    
    for model in model_list:
        result_dict[model] = []
        if not model in target_dict.keys():
            print(f"{model} is not in target_dict")
            continue
        for i in x:
            exp_result = pick_exp_result(exp_dict, model, i, 0, 0, 0, 0)
            index = (
                        "shotnum"
                        + str(i)
                        + "_refineprompt"
                        + str(0)
                        + "_demotext"
                        + str(0)
                        + "_randomdemo"
                        + str(0)
                        + "_maskdemo"
                        + str(0)
                        + "_expid"
                        + str(0)
                    )
            if (not index in target_dict[model].keys()):
                print(f"{model} has no {index} in target_dict")
                result_dict[model].append(0)
                continue
            target_result = target_dict[model][index]
            if (exp_result["error_num"] > exp_result["Total"][1] * filter_threshold or
                target_result["error_num"] > target_result["Total"][1] * filter_threshold):
                result_dict[model].append(0)
            else:
                result_dict[model].append

            if exp_result["error_num"] > exp_result["Total"][1] * filter_threshold:
                result_dict[model].append(0)
            else:
                result_dict[model].append(exp_result["Total"][-1])
    eta_list = []
    delta_list = []
    maxacc_list = []
    for model in model_list:
        y = result_dict[model]
        print(model, y)
        if not "human" in model and y.count(0) > 4:
            print(f"- {model}: exps have not been finished, lack {y.count(0)}")
            eta_list.append(0)
            delta_list.append(0)
            maxacc_list.append(0)
            continue

        print(f"- [{model}] eta: {cal_eta(y)}, delta: {cal_delta(y)}")
        result += f"[{model}]\n - eta: {cal_eta(y)}, delta: {cal_delta(y)}\n"
        eta_list.append(cal_eta(y))
        delta_list.append(cal_delta(y))
        maxacc_list.append(max(y))
    
    return result_dict, eta_list, delta_list, maxacc_list, result

def level_analysis(exp_dict_list):
    exp_dict_level = {"detail":{}, "pattern":{}, "element":{}, "style":{},  "unkown":{}}
    level_mapping_dict = {
        "detail": ["MVTec-AD", "MVTecAD", "MVTec-AD-2", "MVTecAD2_new", "MVTec-LOCO", "VisA", "ITD", "ISP-AD", "WFDD", "AITEX", "RICE-LEAF-DISEASE", "RiceLeaf", "TOMATO-LEAF", "tomatoleaf", "FabricsDefect", ],
        "pattern": ["BTAD", "NEU-DET", "Fabrics", "Fabrics_new", "Frame", "frame", "MeatDataset", "meatDataset"],
        "element": ["plantLeaf", "Product1M", "CosmeticRecognition_new", "SceneryWatermark", "SceneryWatermark_new", "RiceImage", "riceImage", "CAMO", "CPD10K", "COD10K_new", "NC4K", "NC4K_new", "25-Indian-Bird", "IndianBird1", "IndianBird", "IndianBird3", "EPLID", "V2-Plant-Seedlings", "PlantSeedings"],
        "style": ["ImageExposures", "imageExposures", "TransformImage"]
    }
    for i in range(0, len(exp_dict_list)):
        exp_dict = exp_dict_list[i]
        for model in exp_dict.keys():
            for index in exp_dict[model].keys():
                for name in exp_dict[model][index]:
                    if (not "/" in name):
                        continue
                    dataset = name.split("/")[2]
                    level = "unkown"
                    for level_name in level_mapping_dict.keys():
                        if dataset in level_mapping_dict[level_name]:
                            level = level_name
                    if (level == "unkown"):
                        print(f"{dataset} not found in the mapping dict")
                    if (model in exp_dict_level[level].keys()):
                        if(index in exp_dict_level[level][model].keys()):
                            exp_dict_level[level][model][index]["Total"][0] += exp_dict[model][index][name][0]
                            exp_dict_level[level][model][index]["Total"][1] += exp_dict[model][index][name][1]
                        else:
                            exp_dict_level[level][model][index] = {"Total": exp_dict[model][index][name], "error_num": 0}
                    else:
                        exp_dict_level[level][model] = {index: {"Total": exp_dict[model][index][name], "error_num": 0}}
    

    for level in exp_dict_level:
        for model in exp_dict_level[level]:
            for index in exp_dict_level[level][model]:
                if (exp_dict_level[level][model][index]["Total"][1] == 0):
                    exp_dict_level[level][model][index][-1] = 0
                else:
                    exp_dict_level[level][model][index][-1] = round(exp_dict_level[level][model][index]["Total"][0] / exp_dict_level[level][model][index]["Total"][1] ,2)
        print(level, exp_dict_level[level].keys())

    return exp_dict_level
    # level_dict = {}
    # unkown_list = []

    # dataset_dict = {}
    # exp_dict_list = []
    # index_list = []
    # jsonl_file_index = 0
    # for jsonl_file in jsonl_file_list:
    #     exp_dict = {}
    #     with open(jsonl_file, "r") as f:
    #         for line in f.readlines():
    #             data = json.loads(line)
    #             log_path = data["log_path"]
    #             with open(log_path, "r") as f:
    #                 data = json.loads(line)
    #                 if(not data["model"] == "gpt-4o"):
    #                     continue
    #                 for item in data["exp_result"].keys():
    #                     if (not "/" in item):
    #                         continue
    #                     dataset = item.split("/")[2]
    #                     level = "unkown"
    #                     for level_name in level_mapping_dict.keys():
    #                         if dataset in level_mapping_dict[level_name]:
    #                             level = level_name
    #                     if (level == "unkown"):
    #                         print(f"{dataset} not found in the mapping dict")
    #                         if (not dataset in unkown_list):
    #                             unkown_list.append(dataset)
    #                     if (not level in level_dict.keys()):
    #                         level_dict[level] = [0, 0]
    #                     level_dict[level][0] += data["exp_result"][item][0]
    #                     level_dict[level][1] += data["exp_result"][item][1]
                        
    #                     # if (not dataset in dataset_dict.keys()):
    #                     #     dataset_dict[dataset] = [0, 0]
    #                     # dataset_dict[dataset][0] += data["exp_result"][item][0]
    #                     # dataset_dict[dataset][1] += data["exp_result"][item][1]

    # # for dataset in dataset_dict:
    # #     print(dataset, dataset_dict[dataset], round(dataset_dict[dataset][0]/dataset_dict[dataset][1], 2))

    # for level in level_dict:
    #     print(level, level_dict[level], round(level_dict[level][0]/level_dict[level][1], 2))
    # print("unkown_list", unkown_list)




def vis_exp_result_lines(jsonl_file_list):
    """
    根据输入的jsonl文件可视化实验结果。

    Args:
        jsonl_file (str): jsonl格式的实验结果文件路径。

    Returns:
        None

    """

    exp_dict_list = []
    index_list = []
    jsonl_file_index = 0
    for jsonl_file in jsonl_file_list:
        exp_dict = {}
        with open(jsonl_file, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                log_path = data["log_path"]
                model = data["model"]
                shot_num = data["shot_num"]
                refine_prompt_flag = data["refine_prompt_flag"]
                demo_text_flag = data["demo_text_flag"]
                random_demo_flag = data["random_demo_flag"]
                mask_demo_flag = data["mask_demo_flag"]
                exp_result = data["exp_result"]
                index = (
                    "shotnum"
                    + str(shot_num)
                    + "_refineprompt"
                    + str(refine_prompt_flag)
                    + "_demotext"
                    + str(demo_text_flag)
                    + "_randomdemo"
                    + str(random_demo_flag)
                    + "_maskdemo"
                    + str(mask_demo_flag)
                    + "_expid"
                    + str(0)
                )

                # print(shot_num, refine_prompt_flag, demo_text_flag, random_demo_flag)
                if model in exp_dict.keys():
                    if index in exp_dict[model].keys():
                        if (
                            exp_dict[model][index]["error_num"]
                            > exp_result["error_num"]
                        ):
                            exp_dict[model][index] = exp_result
                    else:
                        exp_dict[model][index] = exp_result
                else:
                    exp_dict[model] = {
                        index: exp_result
                    }
                if not index in index_list:
                    index_list.append(index)
            exp_dict_list.append(exp_dict)
    exp_dict_list = exp_dict_list[:-1]

    exp_dict_level = level_analysis(exp_dict_list)
    level_list = list(exp_dict_level.keys())
    exp_dict_list_level = list(exp_dict_level.values())

    human_list = [
        [
            [0, 0, 0, 0, 0, {"Total": [705, 1050, 0.6714], "error_num": 0}],
            [1, 0, 0, 0, 0, {"Total": [919, 1050, 0.8752], "error_num": 0}],
            [1, 0, 0, 2, 0, {"Total": [918, 1050, 0.9180], "error_num": 0}],
            [3, 0, 0, 0, 0, {"Total": [911, 1050, 0.9110], "error_num": 0}],
            [5, 0, 0, 0, 0, {"Total": [906, 1050, 0.9060], "error_num": 0}],
        ],
        [
            [0, 0, 0, 0, 0, {"Total": [634, 1050, 0.6038], "error_num": 0}],
            [1, 0, 0, 0, 0, {"Total": [864, 1050, 0.8229], "error_num": 0}],
            [1, 0, 0, 2, 0, {"Total": [876, 1050, 0.8343], "error_num": 0}],
            [3, 0, 0, 0, 0, {"Total": [902, 1050, 0.8590], "error_num": 0}],
            [5, 0, 0, 0, 0, {"Total": [888, 1050, 0.8457], "error_num": 0}],
        ],
        [
            [0, 0, 0, 0, 0, {"Total": [314, 1050, 0.2990], "error_num": 0}],
            [1, 0, 0, 0, 0, {"Total": [752, 1050, 0.7162], "error_num": 0}],
            [1, 0, 0, 2, 0, {"Total": [819, 1050, 0.7800], "error_num": 0}],
            [3, 0, 0, 0, 0, {"Total": [0, 0, 0], "error_num": 0}],
            [5, 0, 0, 0, 0, {"Total": [0, 0, 0], "error_num": 0}],
        ],
    ]
    for i in range(0, len(human_list)):
        for j in range(0, len(human_list[i])):
            item = human_list[i][j]
            index = (
                "shotnum"
                + str(item[0])
                + "_refineprompt"
                + str(item[1])
                + "_demotext"
                + str(item[2])
                + "_randomdemo"
                + str(item[3])
                + "_maskdemo"
                + str(item[4])
                + "_expid"
                + str(0)
            )
            if "human" in exp_dict_list[i].keys():
                exp_dict_list[i]["human"][index] = item[5]
            else:
                exp_dict_list[i]["human"] = {index: item[5]}

    dataset_list = ["Manufacturing", "E-commence", "Agriculture", "Meidcal"]
    display_model_result(exp_dict_list, dataset_list)
    exp_dict, index_co = combine_model_result(exp_dict_list)

    #exp_dict = exp_dict_list_level[0]

    num_shot = 5
    x = range(0, num_shot + 1)  # x轴坐标
    # exp_dict = exp_dict_list[3]
    model_list = list(exp_dict.keys())
    colors = plt.get_cmap("tab20", len(model_list))
    colors = [colors(i) for i in range(len(model_list))]
    colors_dict = dict(zip(model_list, colors))

    # ** # Tendency of Accuracy vs Shot Numbers
    result_dict = {}
    filter_threshold = 0.8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    result_dict, eta_list, delta_list, maxacc_list, result = cal_eta_delta_from_exp_result(exp_dict, exp_dict, filter_threshold)

    for model in model_list:
        y = result_dict[model]
        new_x = [xi for xi, yi in zip(x, y) if yi != 0]
        new_y = [yi for yi in y if yi != 0]
        if "random" in model:
            ax1.axhline(
                y=average(new_y), color=colors_dict[model], linestyle="--", linewidth=2
            )
        else:
            ax1.plot(
                new_x,
                new_y,
                marker="o",
                color=colors_dict[model],
                linestyle="-",
                label=model,
            )

    ax1.text(
        1.05,
        0.05,
        result,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    # 添加标题和标签
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("Tendency of Accuracy vs Shot Numbers")
    ax1.set_ylabel("Shot Number")
    ax1.set_xlabel("Accuracy")  # 添加图例
    ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    plt.tight_layout()
    # 显示网格
    # plt.grid(True)

    ##
    bubble_sizes = [
        -1 / np.log(acc + 1e-6) * 2000 for acc in maxacc_list
    ]  # 添加小常数以避免log(0)
    scatter = ax2.scatter(
        delta_list,
        eta_list,
        c=colors,
        s=bubble_sizes,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    for i in range(len(eta_list)):
        ax2.annotate(
            model_list[i],
            (delta_list[i], eta_list[i]),
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
            fontsize=9,
        )

    ax2.set_ylim(0.5, max(eta_list) + 0.05)
    ax2.set_xlim(0.1, max(delta_list) + 0.2)
    # 添加标题和轴标签
    ax2.set_title("η vs δ with Bubble Size Representing Max Accuracy")
    ax2.set_ylabel("η: efficiency")
    ax2.set_xlabel("δ: effectiveness")

    save_path = "sample/0430/tendency_accuarcy_vs_shots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)
    print("----------------------------")

    # eta_list_dict = {}
    # delta_list_dict = {}
    # maxacc_list_dict = {}
    # dataset_list = dataset_list[:-1]
    # for i in range(0, len(dataset_list)):
    #     sub_exp_dict = exp_dict_list[i]
    #     result_dict, eta_list_dict[dataset_list[i]], delta_list_dict[dataset_list[i]], maxacc_list_dict[dataset_list[i]], result = cal_eta_delta_from_exp_result(sub_exp_dict, exp_dict, filter_threshold)

    # draw_radar_picture(dataset_list, model_list, colors_dict, maxacc_list_dict, "macacc")


    # eta_list_dict = {}
    # delta_list_dict = {}
    # maxacc_list_dict = {}
    # dataset_list = level_list
    # for i in range(0, len(dataset_list)):
    #     sub_exp_dict = exp_dict_list_level[i]
    #     result_dict, eta_list_dict[dataset_list[i]], delta_list_dict[dataset_list[i]], maxacc_list_dict[dataset_list[i]], result = cal_eta_delta_from_exp_result(sub_exp_dict, exp_dict, filter_threshold)

    # draw_radar_picture(dataset_list, model_list, colors_dict, maxacc_list_dict, "macacc")
    # print(maxacc_list_dict)



    drawing_dict_cause = {
        "w. mask + enhance": [0, 0, 0, -1],
        "w. detailed instruction": [2, 0, 0, 0],
        "w. caption demo": [0, 1, 0, 0],
    }
    draw_line_picture(
        num_shot,
        exp_dict,
        colors_dict,
        drawing_dict_cause,
        "cause_location",
        filter_threshold,
    )
    drawing_dict_mechanism = {
        "+ fabricate demo": [0, 0, 2, 0],
        "+ replicate demo": [0, 0, 6, 0],
        "+ negative noisy demo": [0, 0, 5, 0],
    }
    draw_line_picture(
        num_shot,
        exp_dict,
        colors_dict,
        drawing_dict_mechanism,
        "machanism_analysis",
        filter_threshold,
    )

def draw_radar_picture(dataset_list, model_list, colors_dict, data_list_dict, text):
    # Define the labels for each axis
    labels = dataset_list
    num_vars = len(labels)
    # Compute angle for each axis in the plot (in radians)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i in range(len(model_list)):
        model = model_list[i]
        datas = []
        for label in labels:
            datas.append(data_list_dict[label][i])

        values = datas
        #print(angles, values)
        # Draw the outline of the radar chart
        values += values[:1]
        ax.plot(angles, values, color=colors_dict[model], label = model, linewidth=1)
        ax.fill(angles, values, color=colors_dict[model], alpha=0.25)

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set the range for the radial axis
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    # Add a title
    plt.title(f'Radar Chart Example of {text}', size=20, y=1.05)

    save_path = "sample/0430/radar_of_effciency_effectiveness.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)


def draw_line_picture(
    num_shot, exp_dict, colors_dict, drawing_dict, exp_name, filter_threshold=1
):
    """
    绘制线条图以展示不同模型的性能对比。

    Args:
        num_shot (int): 实验中使用的样本数量。
        exp_dict (dict): 存储实验结果的字典，键为模型名称，值为实验结果列表。
        colors_dict (dict): 存储模型对应颜色的字典，键为模型名称，值为颜色代码。
        flag (tuple): 包含四个整数的元组，用于指定从实验结果中挑选哪些值。
        text (str): 附加到图表的文本说明。

    Returns:
        None

    """
    x = range(0, num_shot + 1)  # x轴坐标
    model_list = list(exp_dict.keys())
    # ** #
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(drawing_dict.keys()),
        figsize=(28, 10),
        sharex=False,
        sharey=False,
    )
    for index in range(0, len(drawing_dict.keys())):
        # print(index, len(drawing_dict.keys()))
        ax = axes[index]
        text = list(drawing_dict.keys())[index]
        flag = drawing_dict[text]
        print(f"---{text}---{flag}")
        result_dict = {}
        for model in model_list:
            result_dict[model] = []
            for i in x:
                exp_result = pick_exp_result(exp_dict, model, i, 0, 0, 0, 0)
                if exp_result["error_num"] > exp_result["Total"][1] * filter_threshold:
                    result_dict[model].append(0)
                else:
                    result_dict[model].append(exp_result["Total"][-1])
        result_dict_new = {}
        for model in model_list:
            result_dict_new[model] = []
            for i in x:
                exp_result = pick_exp_result(
                    exp_dict, model, i, flag[0], flag[1], flag[2], flag[3]
                )
                if exp_result["error_num"] > exp_result["Total"][1] * filter_threshold:
                    result_dict_new[model].append(0)
                else:
                    result_dict_new[model].append(exp_result["Total"][-1])
        result = ""
        for model in model_list:
            y1 = result_dict[model]
            y2 = result_dict_new[model]
            if "random" in model:
                new_y = [yi for yi in y1 + y2 if yi != 0]
                ax.axhline(
                    y=average(new_y),
                    color=colors_dict[model],
                    linestyle="--",
                    linewidth=2,
                )
                continue
            if y1.count(0) > 4 or y2.count(0) > 4:
                print(
                    f"- {model}: exps have not been finished, lack {y1.count(0)} and {y2.count(0)}"
                )
                continue

            print(f"- [{model}] eta: {cal_phi(y2, y1)}")
            result += f"- [{model}] eta: {cal_phi(y2, y1)}\n"
            new_x = [xi for xi, yi in zip(x, y1) if yi != 0]
            new_y = [yi for yi in y1 if yi != 0]
            ax.plot(
                new_x,
                new_y,
                marker="o",
                color=colors_dict[model],
                linestyle="--",
                label=model,
            )
            new_x = [xi for xi, yi in zip(x, y2) if yi != 0]
            new_y = [yi for yi in y2 if yi != 0]
            ax.plot(
                new_x,
                new_y,
                marker="x",
                color=colors_dict[model],
                linestyle="-",
                label=model + " " + text,
            )

        ax.text(
            0,
            -0.2,
            result,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
        )

        # 添加标题和标签
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("Analysis on " + text)
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Accuracy")  # 添加图例
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=1, borderaxespad=0.0
        )
        plt.tight_layout()
    # 显示网格
    # plt.grid(True)
    save_path = "sample/0430/tendency_combine_" + exp_name + ".png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)
    print("----------------------------")


def vis_exp_result(jsonl_file):
    """
    可视化实验结果的函数。

    Args:
        jsonl_file (str): 包含实验结果的JSONL文件的路径。

    Returns:
        None

    """
    exp_dict = {}
    index_list = []
    with open(jsonl_file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            log_path = data["log_path"]
            model = data["model"]
            shot_num = data["shot_num"]
            refine_prompt_flag = data["refine_prompt_flag"]
            demo_text_flag = data["demo_text_flag"]
            random_demo_flag = data["random_demo_flag"]
            mask_demo_flag = data["mask_demo_flag"]
            exp_result = data["exp_result"]
            index = (
                "shotnum"
                + str(shot_num)
                + "_refineprompt"
                + str(refine_prompt_flag)
                + "_demotext"
                + str(demo_text_flag)
                + "_randomdemo"
                + str(random_demo_flag)
                + "_maskdemo"
                + str(mask_demo_flag)
                + "_expid"
                + str(0)
            )
            if "intern" in model:
                print(key, exp_result["Total"], exp_result["error_num"])

            # print(shot_num, refine_prompt_flag, demo_text_flag, random_demo_flag)
            if model in exp_dict.keys():
                if index in exp_dict[model].keys():
                    if exp_dict[model][index]["error_num"] > exp_result["error_num"]:
                        exp_dict[model][index] = exp_result
                else:
                    exp_dict[model][index] = exp_result
            else:
                exp_dict[model] = {index: exp_result}
            if not index in index_list:
                index_list.append(index)

    # for key in exp_dict.keys():
    #    print(key, exp_dict[key].keys())
    # print(index_list)
    # model_list.remove("ernie-4.5-8k-preview")

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    # plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

    figsize = (36, 6)
    model_list = list(exp_dict.keys())
    labels = model_list
    x = np.arange(len(labels))  # 标签位置
    width = 0.2  # 柱状图的宽度
    fig, ax = plt.subplots(figsize=figsize)

    ## Draw Pic: Model Comparison with Visual ICL for random demo
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    group6 = []
    for model in model_list:
        group1.append(pick_exp_result(exp_dict, model, 0, 0, 0, 0)["Total"][-1])
        group2.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0)["Total"][-1])
        group3.append(pick_exp_result(exp_dict, model, 2, 0, 0, 0)["Total"][-1])
        group4.append(pick_exp_result(exp_dict, model, 3, 0, 0, 0)["Total"][-1])
        group5.append(pick_exp_result(exp_dict, model, 4, 0, 0, 0)["Total"][-1])
        group6.append(pick_exp_result(exp_dict, model, 5, 0, 0, 0)["Total"][-1])

    x = np.arange(len(labels))  # 标签位置
    width = 0.12  # 柱状图的宽度
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width * 5 / 2, group1, width, label="zero shot")
    rects2 = ax.bar(x - width * 3 / 2, group2, width, label="one shot")
    rects3 = ax.bar(x - width * 1 / 2, group3, width, label="two shot")
    rects4 = ax.bar(x + width * 1 / 2, group4, width, label="three shot")
    rects5 = ax.bar(x + width * 3 / 2, group5, width, label="four shot")
    rects6 = ax.bar(x + width * 5 / 2, group6, width, label="five shot")

    ax.set_ylabel("acc scores")
    ax.set_title("Model Comparison with Visual ICL for various shots")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects3:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects4:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects5:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects6:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # fig.tight_layout()

    save_path = "sample/0416/exp_vis_cleanprompt_variousshot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)

    ## Draw Pic: Model Comparison with Visual ICL for random demo
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    group6 = []
    for model in model_list:
        group1.append(pick_exp_result(exp_dict, model, 0, 0, 0, 0, 0)["Total"][-1])
        group2.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0, 0)["Total"][-1])
        group3.append(pick_exp_result(exp_dict, model, 1, 0, 0, 1, 0)["Total"][-1])
        group4.append(pick_exp_result(exp_dict, model, 1, 0, 0, 2, 0)["Total"][-1])
        group5.append(pick_exp_result(exp_dict, model, 1, 0, 0, 4, 0)["Total"][-1])
        group6.append(pick_exp_result(exp_dict, model, 1, 0, 0, 5, 0)["Total"][-1])
        # print("**",model, pick_exp_result(exp_dict, model, 1, 0, 0, 5, 0))

    x = np.arange(len(labels))  # 标签位置
    width = 0.12  # 柱状图的宽度
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width * 5 / 2, group1, width, label="zero shot")
    rects2 = ax.bar(x - width * 3 / 2, group2, width, label="one shot with good label")
    rects3 = ax.bar(
        x - width * 1 / 2, group3, width, label="one shot with exchange label"
    )
    rects4 = ax.bar(
        x + width * 1 / 2, group4, width, label="one shot with fabrictae label"
    )
    rects5 = ax.bar(
        x + width * 3 / 2, group5, width, label="one shot with total noise demo image"
    )
    rects6 = ax.bar(
        x + width * 5 / 2,
        group6,
        width,
        label="one shot with negative noise demo image",
    )

    ax.set_ylabel("acc scores")
    ax.set_title("Model Comparison with Visual ICL for random demo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects3:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects4:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects5:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects6:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # fig.tight_layout()

    save_path = "sample/0416/exp_vis_cleanprompt_randomdemo.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)

    ## Draw Pic: Model Comparison with Visual ICL for various prompt
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    for model in model_list:
        group1.append(pick_exp_result(exp_dict, model, 0, 0, 0, 0, 0)["Total"][-1])
        group2.append(pick_exp_result(exp_dict, model, 0, 1, 0, 0, 0)["Total"][-1])
        group3.append(pick_exp_result(exp_dict, model, 0, 2, 0, 0, 0)["Total"][-1])
        group4.append(pick_exp_result(exp_dict, model, 0, 0, 0, 0, -1)["Total"][-1])
        # print("**", model, pick_exp_result(exp_dict, model, 0, 2, 0, 0, 0)["Total"])

    x = np.arange(len(labels))  # 标签位置
    width = 0.12  # 柱状图的宽度
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(
        x - width * 3 / 2, group1, width, label="zero shot with clean prompt"
    )
    rects2 = ax.bar(x - width * 1 / 2, group2, width, label="zero shot with cot prompt")
    rects3 = ax.bar(
        x + width * 1 / 2, group3, width, label="zero shot with detailed prompt"
    )
    rects4 = ax.bar(
        x + width * 3 / 2, group4, width, label="zero shot with attention mask"
    )

    ax.set_ylabel("acc scores")
    ax.set_title("Model Comparison with Zero shot with various instruction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects3:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects4:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # fig.tight_layout()

    save_path = "sample/0416/exp_vis_zeroshot_variousprompt.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)

    ## Draw Pic: Model Comparison with Visual ICL for random demo
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    for model in model_list:
        group1.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0, 0)["Total"][-1])
        group2.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0, 3)["Total"][-1])
        group3.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0, 2)["Total"][-1])
        group4.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0, 1)["Total"][-1])
        group5.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0, -1)["Total"][-1])

    x = np.arange(len(labels))  # 标签位置
    width = 0.12  # 柱状图的宽度
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width * 4 / 2, group1, width, label="one shot")
    rects2 = ax.bar(x - width * 2 / 2, group2, width, label="one shot with 0.33 mask")
    rects3 = ax.bar(x, group3, width, label="one shot with 0.50 mask")
    rects4 = ax.bar(x + width * 2 / 2, group4, width, label="one shot with 1.00 mask")
    rects5 = ax.bar(
        x + width * 4 / 2,
        group5,
        width,
        label="one shot with enhenced scaling detail query image",
    )

    ax.set_ylabel("acc scores")
    ax.set_title("Model Comparison with Visual ICL for random demo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects3:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects4:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects5:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # fig.tight_layout()

    save_path = "sample/0416/exp_vis_oneshot_variousmask.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)

    ## Draw Pic: Model Comparison with Visual ICL for demo info (clean prompt)
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    for model in model_list:
        group1.append(pick_exp_result(exp_dict, model, 0, 0, 0, 0)["Total"][-1])
        group2.append(pick_exp_result(exp_dict, model, 1, 0, 0, 0)["Total"][-1])
        group3.append(pick_exp_result(exp_dict, model, 1, 0, 1, 0)["Total"][-1])
        group4.append(pick_exp_result(exp_dict, model, 1, 0, 2, 0)["Total"][-1])

    x = np.arange(len(labels))  # 标签位置
    width = 0.2  # 柱状图的宽度
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width * 3 / 2, group1, width, label="zero shot")
    rects2 = ax.bar(x - width * 1 / 2, group2, width, label="one shot with label")
    rects3 = ax.bar(x + width * 1 / 2, group3, width, label="one shot with caption")
    rects4 = ax.bar(
        x + width * 3 / 2, group4, width, label="one shot with visual_intro"
    )

    ax.set_ylabel("acc scores")
    ax.set_title("Model Comparison with Visual ICL for demo info (clean prompt)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects3:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects4:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # fig.tight_layout()

    save_path = "sample/0416/exp_vis_cleanprompt_varioudemo.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)

    ## Draw Pic: Model Comparison with Visual ICL for demo info (cot prompt)
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    for model in model_list:
        group1.append(pick_exp_result(exp_dict, model, 0, 1, 0, 0)["Total"][-1])
        group2.append(pick_exp_result(exp_dict, model, 1, 1, 0, 0)["Total"][-1])
        group3.append(pick_exp_result(exp_dict, model, 1, 1, 1, 0)["Total"][-1])
        group4.append(pick_exp_result(exp_dict, model, 1, 1, 2, 0)["Total"][-1])

    x = np.arange(len(labels))  # 标签位置
    width = 0.2  # 柱状图的宽度
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width * 3 / 2, group1, width, label="zero shot")
    rects2 = ax.bar(x - width * 1 / 2, group2, width, label="one shot with label")
    rects3 = ax.bar(x + width * 1 / 2, group3, width, label="one shot with caption")
    rects4 = ax.bar(
        x + width * 3 / 2, group4, width, label="one shot with visual_intro"
    )

    ax.set_ylabel("acc scores")
    ax.set_title("Model Comparison with Visual ICL for demo info (cot prompt)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects3:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    for rect in rects4:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # fig.tight_layout()

    save_path = "sample/0416/exp_vis_cotprompt_varioudemo.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print("finish ", save_path)


def vis(jsonl_file):
    """
    从jsonl文件中随机抽取20个样本并可视化显示。

    Args:
        jsonl_file (str): 包含样本数据的jsonl文件路径。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果提供的jsonl_file路径不存在，将抛出此异常。
        json.JSONDecodeError: 如果jsonl文件中的某一行数据格式不正确，
            无法解析为json，将抛出此异常。
    """

    sample_list = []
    with open(jsonl_file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            if not "shot_list" in data:
                continue
            sample_list.append(data)

    sample_list = random.sample(sample_list, k=20)
    for data in sample_list:
        img_path = data["img_path"]
        shot_list = data["shot_list"]
        label = data["label"]
        prompt = data["prompt"]
        result = data["result"]
        img = Image.open(img_path)
        images = [img] + [Image.open(path[0]) for path in shot_list]

        cols = 3  # 每行显示3张图
        rows = int(len(shot_list) / cols) + 1
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # 动态调整画布高度

        for i, (img, ax) in enumerate(zip(images, axes.flat)):
            ax.imshow(img)
            if i == 0:
                ax.set_title(
                    f"Query Image \n with gt as {label}\n with doubao_result as {result}",
                    fontsize=20,
                )
            else:
                ax.set_title(
                    f"Example Image {i}\n{shot_list[i-1][1]}", fontsize=20
                )  # 标题含文件名
            ax.axis("off")

        fig.suptitle(
            f"Instruct: {prompt}",
            y=1.02,  # 垂直位置微调（数值越大越靠上）
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        save_path = "sample/0411/vis_" + img_path.replace("/", "_")
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", pad_inches=0.1  # 提高分辨率  # 去除多余白边
        )  # 保留少量边距
        print("finish ", save_path)


if __name__ == "__main__":

    # img_path = "/mnt/cfs_bj/liannan/visualicl_raw_data/Manufacture/MVTec-AD/pill/test/faulty_imprint/000.png"
    # encode_image_withmask(img_path, mask_demo_flag=-1)

    # jsonl_file = "/mnt/cfs_bj/liannan/visualicl_logs/processed_data/stage2/dataset/2025-04-24/Manufacture_manufacture-final_0.21_4_1050_11:54:51.476301_each_5_shots_demotextflag2_finepromptflag2_randomedemoflag4_maskdemoflag1.jsonl"
    # reasoning_exp_process(jsonl_file)

    ## Convert jsonl to csv (for outside human labeler)
    # jsonl_file = (
    #     "/mnt/cfs_bj/liannan/visualicl_logs/processed_data/stage2/dataset/"
    #     + "2025-04-24/Manufacture_manufacture-final_0.21_4_1050_"
    #     + "11:54:51.476301_each_5_shots_demotextflag2_finepromptflag2_randomedemoflag4_maskdemoflag1.jsonl"
    # )
    # save_file = "./src/images/new_Manufacture_manufacture-final_0.21_4_1050_11:54:51.476301_human.csv"
    # convert_jsonl_to_csv(jsonl_file, save_file)

    ## reorganize_dataset to our certain format
    # reorganize_dataset("/mnt/cfs_bj/liannan/visualicl_raw_data/Manufacture/MVTec-3D/")
    # convert_bmp2png_all("/mnt/cfs_bj/liannan/visualicl_raw_data/Manufacture/MVTec-3D/")

    ## Visualization of some case with demos in the same image
    # jsonl_file = (
    #    "log/stage1/log_2025-04-11/23:17:59.871174_Pattern_Manufature_"
    #    + "NEU-DETsteelstripsRecogization_all_doubao-1.5-vision-pro-32k-250115_500samples_0shot_ _auto.jsonl"
    # )
    # vis(jsonl_file)

    ## Generate Random Images
    # generate_random_img()

    # Visualization of the result
    jsonl_file = [
        (
            "/mnt/cfs_bj/liannan/visualicl_logs/log/stage3/success_exp/"
            + "success_exp_record_manufacture-final_0424.jsonl"
        ),
        (
            "/mnt/cfs_bj/lihuaiming/visualicl_logs/log/stage3/success_exp/"
            + "success_exp_record_agriculture-all_0430.jsonl"
        ),
        (
            "/mnt/cfs_bj/lihuaiming/visualicl_logs/log/stage3/success_exp/"
            + "success_exp_record_E-commerce-all_0503.jsonl"
        ),
        (
            "/mnt/cfs_bj/yuetianyuan/visualicl_logs/log/stage3/success_exp/"
            + "success_exp_record_medical-final_05050505.jsonl"
        ),
    ]
    # parrallel_exp_jsonl = (
    #     "/mnt/cfs_bj/liannan/visualicl_logs/log/stage3/success_exp/"
    #     + "success_exp_record_manufacture-final_0429.jsonl"
    # )
    # comparsion_among_exps(jsonl_file, parrallel_exp_jsonl)
    vis_exp_result_lines(jsonl_file)

    # data_jsonl_file = (
    #     "/mnt/cfs_bj/liannan/visualicl_logs/processed_data/stage2/dataset/2025-04-24/"
    #     + "Manufacture_manufacture-final_0.21_4_1050_11:54:51.476301_"
    #     + "each_5_shots_demotextflag2_finepromptflag2_randomedemoflag4.jsonl"
    # )
    # process_jsonl_path(data_jsonl_file)
