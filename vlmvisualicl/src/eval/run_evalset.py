# run_evalset.py
"""
This file contains the code for running evaluation on a set of images.
"""
import random, json, threading, copy
from threading import Thread, Lock
from utils.helper import (
    encode_image,
    encode_image_withmask,
    prompt_assemble,
    read_filename_from_dir,
    read_filename_from_jsonl,
    save_log,
    cal_metric,
    split_list,
)
from utils.api_call import (
    mainland_vlm_api_call,
    oversea_vlm_api_call,
    siliconflow_vlm_api_call,
)
from utils.api_call import (
    refine_prompt,
    create_detail_demonstration,
    vlm_model_api_gate,
)
from datetime import datetime

data_count = 0


def inference_result_from_picdir_with_thread(
    thread_id,
    lock,
    cate_result_list,
    error_count_list,
    data_list,
    dir,
    industry,
    level,
    model,
    log_path,
    shot_num,
    refine_prompt_flag,
    demo_text_flag,
    random_demo_flag,
    mask_demo_flag,
    cate,
    prompt,
):
    """
    创建详细演示，包含提示、图片列表和文本。

    Args:
        prompt (str): 提示信息，例如"You are an image detail observer."
        shot_list_base64 (List[Tuple[str, str, str]]): 包含三个元素的元组列表，分别为键值（字符串）、图片URL（字符串）和文本（字符串）。
            每一个元组都应该是以下格式：('key', 'url', 'text')。

    Returns:
        List[Tuple[str, str, str]]: 返回一个包含三个元素的元组列表，分别为键值（字符串）、图片URL（字符串）和详细演示（字符串）。

    Raises:
        None
    """
    # print(f"#### no.{thread_id} thread running...")
    cate_result_dict = {}
    id = 1
    error_count = 0
    for data in data_list:
        shot_list = []
        if not "jsonl" in dir:
            img_path = data
            label = img_path.split("/")[-2]
            if not label in cate:
                label = img_path.split("/")[-3]
            candidate = cate
        else:
            if not "query_info" in data.keys():
                continue

            if random_demo_flag == 2 and "fabricated_query_info" in data.keys():
                query_info = data["fabricated_query_info"]
            else:
                query_info = data["query_info"]
            if mask_demo_flag == 0:
                img_path = query_info["img_path"]
            elif mask_demo_flag == -1:
                img_path = query_info["img_path"]

            elif mask_demo_flag > 0:
                index = "-mask-s" + str(mask_demo_flag)
                if "masked_path" in query_info:
                    # print(query_info["masked_path"].keys())
                    if index in query_info["masked_path"]:
                        img_path = query_info["masked_path"][index]
                    else:
                        print(
                            "#####0 no masked img path found, skip",
                            query_info["img_path"],
                        )
                        continue
                else:
                    print(
                        "#####1 no masked img path found, skip", query_info["img_path"]
                    )
                    continue
            else:
                if "masked_path" in query_info:
                    img_path = query_info["img_path"]
                else:
                    print(
                        "#####2 no masked img path found, skip", query_info["img_path"]
                    )
                    continue

            label = query_info["label"]

            candidate = query_info["candidate"]
            if type(candidate) == type("abc"):
                candidate = json.loads(candidate.replace("'", '"'))

            if refine_prompt_flag == 1 and "cot" in query_info["instruct"].keys():
                prompt = query_info["instruct"]["cot"]
            elif refine_prompt_flag == 2 and "detail" in query_info["instruct"].keys():
                prompt = query_info["instruct"]["detail"]
            else:
                prompt = query_info["instruct"]["clean"]

            if random_demo_flag == 0:
                for i in range(0, shot_num):
                    for text in data["demo_info"].keys():
                        shot_list.append(data["demo_info"][text][i])
            elif random_demo_flag == 1:
                for i in range(0, shot_num):
                    for text in data["random_demo_info"].keys():
                        shot_list.append(data["random_demo_info"][text][i])
                label = data["random_query_info"]["random_label"]
            elif random_demo_flag == 2:
                for i in range(0, shot_num):
                    for text in data["fabricated_demo_info"].keys():
                        shot_list.append(data["fabricated_demo_info"][text][i])
            elif random_demo_flag == 3:
                for i in range(0, shot_num):
                    for text in data["blank_demo_info"].keys():
                        shot_list.append(data["blank_demo_info"][text][i])
            elif random_demo_flag == 4:
                for i in range(0, shot_num):
                    for text in data["noise_demo_info"].keys():
                        shot_list.append(data["noise_demo_info"][text][i])
            elif random_demo_flag == 5:
                label = data["query_info"]["label"]
                for i in range(0, shot_num):
                    shot_list.append(data["noise_demo_info"][label][i])
                    for text in data["noise_demo_info"].keys():
                        if (not text == label):
                            shot_list.append(data["noise_demo_info"][text][i])
            elif random_demo_flag == 6:
                for i in range(0, shot_num):
                    for text in data["demo_info"].keys():
                        temp = copy.deepcopy(data["demo_info"][text][i])
                        temp["shot_path"] = data["demo_info"][text][0]["shot_path"]
                        shot_list.append(temp)
                        
            else:
                print(f"[Error] random_demo_flag not found with {random_demo_flag}")

        index = "/".join(img_path.split("raw_data")[-1].split("/")[0:-3])
        if not index in cate_result_dict.keys():
            cate_result_dict[index] = [0, 0]

        json_object = run_for_once(
            thread_id,
            img_path,
            prompt,
            candidate,
            shot_list,
            model,
            label,
            demo_text_flag,
            mask_demo_flag,
        )

        mark = json_object["mark"]
        label = json_object["label"]
        result = json_object["result"]
        json_object["id"] = str(thread_id) + "_" + str(id)

        if not "jsonl" in dir:
            save_object = {
                "data_info": {
                    "id": str(thread_id) + "_" + str(id),
                    "industry": industry,
                    "level": level,
                    "source": img_path.split("raw_data/")[-1].split("/")[1],  # dataset
                },
                "query_info": {
                    "img_path": img_path,
                    "label": label,
                    "candidate": candidate,
                    "instruct": {"clean": prompt},
                },
                "result_list": [[str(datetime.now()), log_path, model, result, mark]],
            }
        else:
            if len(shot_list) > 0:
                save_shot_dict = []
                for shot in shot_list:
                    if demo_text_flag == 0:
                        save_shot_dict.append([shot["shot_path"], shot["label_text"]])
                    elif demo_text_flag == 1:
                        save_shot_dict.append([shot["shot_path"], shot["caption"]])
                    elif demo_text_flag == 2:
                        save_shot_dict.append([shot["shot_path"], shot["visual_intro"]])
            else:
                save_shot_dict = shot_list

            save_object = {
                "timestamp": str(datetime.now()),
                "data_info": data["data_info"],
                "model": model,
                "exp_info": {
                    "shot_num": shot_num,
                    "refine_prompt_flag": refine_prompt_flag,
                    "demo_text_flag": demo_text_flag,
                    "random_demo_flag": random_demo_flag,
                },
                "img_path": img_path,
                "prompt": prompt,
                "candidate": candidate,
                "shot_list": save_shot_dict,
                "label": label,
                "result": result,
                "mark": mark,
            }
        global data_count
        with lock:
            data_count += 1
        if not type(mark) == type(False):
            print(f"***{mark}, result:{result}")
            error_count += 1
        else:
            cate_result_dict[index][1] += 1
            if mark:
                cate_result_dict[index][0] += 1

        save_log(log_path, save_object)

        if id % 10 == 0:
            print(
                f"#### {thread_id} Thread Process... {id} samples, with {error_count} errors, for {model}"
            )
            cal_metric(cate_result_dict)
    cate_result_list[thread_id] = cate_result_dict
    error_count_list[thread_id] = error_count
    return cate_result_dict


def inference_result_from_picdir(
    dir,
    task_set,
    model,
    log_path,
    success_path,
    sample_num=500,
    shot_num=0,
    focus_dataset="",
    refine_prompt_flag=0,
    demo_text_flag=0,
    random_demo_flag=0,
    mask_demo_flag=0,
):
    """
    从指定目录中读取图片数据并进行推理。

    Args:
        dir (str): 图片数据所在的目录路径或包含图片数据信息的jsonl文件路径。
        task_set (dict): 任务设置信息，包含行业、场景、任务、类别、维度和定义等字段。
        model (str): 使用的模型名称。
        log_path (str): 日志文件路径。
        sample_num (int, optional): 样本数量，默认为500。
        shot_num (int, optional): 示例数量，默认为0。
        focus_dataset (str, optional): 关注的数据集名称，默认为空字符串。
        refine_prompt_flag (int, optional): 是否对提示进行细化，默认为0（不进行细化）。
        demo_text_flag (int, optional): 是否包含示例文本，默认为0（不包含）。

    Returns:
        None

    Raises:
        None

    """
    
    lock = Lock()
    if not "jsonl" in dir:
        print(f"### Inference from Dir {dir}, totally {sample_num} samples")

        industry = task_set["industry"]
        scene = task_set["scene"]
        task = task_set["task"]
        cate = task_set["cate"]
        level = task_set["level"]
        definition = task_set["definition"]

        data_list, cate_list = read_filename_from_dir(dir)
        if len(data_list) < 1:
            print("[Error] no file found! Please check the path name", dir)
            return {"exp_result": {"Total": [0, 0]}}

        if len(cate) < 2:
            cate = cate_list
        else:
            cate = cate
        prompt_template = (
            "## Role\n You are an expert in {{industry}} industry and ready for the task of {{scene}}. "
            + "You should look carefully the {{level}} level information, noted as {{definition}}. \n"
            + " ## Task \n You task is {{task}}. The potential categories are {{cate}}, \n"
            + "##Constrain\n Just give me the final aswer and no other explain"
        )

        prompt = prompt_assemble(prompt_template, "industry", industry)
        prompt = prompt_assemble(prompt, "scene", scene)
        prompt = prompt_assemble(prompt, "level", level)
        prompt = prompt_assemble(prompt, "definition", definition)
        prompt = prompt_assemble(prompt, "task", task)
        prompt = prompt_assemble(prompt, "cate", str(cate))

    else:
        data_list = []
        with open(dir, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())
                if "query_info" in data.keys():
                    data_list.append(data)

        if not focus_dataset == "all":
            data_list = [
                item
                for item in data_list
                if (focus_dataset in item["query_info"]["label"])
            ]
        prompt = ""
        industry = ""
        level = ""
        cate = ""
    
    succeed_list = []
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            if "query_info" in data.keys():
                succeed_list.append(data["query_info"]["img_path"])
    
    if not "jsonl" in dir:
        data_list = [item for item in data_list if not item in succeed_list]
    else:
        data_list = [item for item in data_list if not item["query_info"]["img_path"] in succeed_list]
    sample_num = sample_num - len(succeed_list)
    print(f"### actually using {len(data_list)} data, with {len(succeed_list)} loaded")

    if sample_num < len(data_list):
        print(
            f"### Checking... \n sampling {sample_num} data from dataset size  of {len(data_list)}"
        )
        data_list = random.sample(data_list, sample_num)
    else:
        print(
            f"### Checking... \n dataset size {len(data_list)} is smaller than sample num{sample_num}"
        )
    if "ernie" in model:
        threads_num = 2
    elif "Qwen2.5-VL-72B-Instruct" in model:
        threads_num = 30
    elif "gemini" in model:
        threads_num = 60
    else:
        threads_num = 30
    if len(data_list) < threads_num:
        threads_num = len(data_list)
    print(f"### Beginning to start {threads_num} threads with {len(data_list)} data")

    cate_result_list = [None] * threads_num
    error_count_list = [None] * threads_num

    data_split_list = split_list(data_list, threads_num)

    threads = []
    for i in range(0, threads_num):  # 创建5个线程
        t = threading.Thread(
            target=inference_result_from_picdir_with_thread,
            args=(
                i,
                lock,
                cate_result_list,
                error_count_list,
                data_split_list[i],
                dir,
                industry,
                level,
                model,
                log_path,
                shot_num,
                refine_prompt_flag,
                demo_text_flag,
                random_demo_flag,
                mask_demo_flag,
                cate,
                prompt,
            ),
        )
        threads.append(t)
        t.start()

    global data_count
    while any(thread.is_alive() for thread in threads):
        with lock:
            print(
                f"Progress: {data_count}/{len(data_list)} ({data_count/len(data_list):.2%}), {model}",
                end="\r",
            )

    for t in threads:
        t.join()  # 等待所有线程完成

    cate_result_dict = {}
    error_count = 0
    # print(cate_result_list)
    # print(error_count_list)
    for cate_result in cate_result_list:
        for key in cate_result.keys():
            if key in cate_result_dict.keys():
                cate_result_dict[key][0] += cate_result[key][0]
                cate_result_dict[key][1] += cate_result[key][1]
            else:
                cate_result_dict[key] = [cate_result[key][0], cate_result[key][1]]

    for item in error_count_list:
        error_count += item

    result = cal_metric(cate_result_dict)
    result["error_num"] = error_count
    save_log(log_path, result)
    print(f"## Finish Inference, log is saving to {log_path} with {error_count} errors")

    return result


def run_for_once(
    thread_id, img_path, prompt, candidate, shot_list, model, label, demo_text_flag, mask_demo_flag=0
):
    """
    从调用一次模型API。

    Args:
        img_path (str): 图片数据所在文件路径。
        prompt (str): 任务指令。
        shot_info (list): 需要fewshot的列表
        model (str): 使用的模型名称。
        label (str): 样本数量，默认为500。
        demo_text_flag (int): 是否包含示例文本，默认为0（不包含）。

    Returns:
        None

    Raises:
        None

    """
    url = encode_image_withmask(img_path, mask_demo_flag)
    shot_list_base64 = []

    temp = ""
    for i in range(0, len(shot_list)):
        shot = shot_list[i]
        shot_path = shot["shot_path"]
        new_url = encode_image(shot_path)
        if demo_text_flag == 0:
            text = shot["label_text"]
        elif demo_text_flag == 1:
            if "caption" in shot.keys():
                text = shot["caption"]
            else:
                print(
                    f"[Warning] found one shot without caption so skip {shot_path} with {len(shot_list)} demos in total"
                )
                continue
        elif demo_text_flag == 2:
            prefix = shot["visual_intro"][0]
            postfix = shot["visual_intro"][1]
            if i == 0:
                prompt = prompt + "\n" + prefix
                text = postfix
            elif i == len(shot_list) - 1:
                shot_list_base64[i - 1][2] = shot_list_base64[i - 1][2] + "\n" + prefix
                text = postfix + "\n Please answer the category of [query_image]"
            else:
                shot_list_base64[i - 1][2] = shot_list_base64[i - 1][2] + "\n" + prefix
                text = postfix

        else:
            print(f"[Error] demo_text_flag not found! {demo_text_flag}")
            return

        shot_list_base64.append([shot_path, new_url, text])

    if "random" in model:
        result = candidate[random.SystemRandom().randint(0, len(candidate) - 1)]
    else:
        result = vlm_model_api_gate(
            prompt, shot_list_base64, url, model, retry=2, thread_id=thread_id
        )

    if "Error" in result or "error" in result:
        mark = "Error"
    else:
        if "<answer>" in result:
            result = result.split("<answer>")[1]
        if "</thinking>" in result:
            result = result.split("</thinking>")[-1]
        if "</think>" in result:
            result = result.split("</think>")[-1]
        if len(result) > 500:
            print(
                f"[Warning] the result len is too long : {len(result)}, please check, {result}"
            )
        mark = label in result
        if "provide" in result:
            print(f"[Error] doesn't found the image : {result}")
            mark = "Error"

    shot_list = []
    for item in shot_list_base64:
        shot_list.append([item[0], item[-1]])

    json_object = {
        "id": str(id),
        "prompt": prompt,
        "shot_list": shot_list,
        "img_path": img_path,
        "result": result,
        "label": label,
        "mark": mark,
    }

    return json_object
