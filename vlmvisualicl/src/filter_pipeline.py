# filter_pipeline
"""
this file is used to process the raw data
"""
import json, os
from utils.helper import (
    save_log,
    read_filename_from_dir,
    random_sample_fewshot,
    average_update,
)
from datetime import datetime
import argparse, random, string, copy, threading
from utils.feature_extraction import sample_by_diversity
from utils.api_call import refine_prompt, create_detail_demonstration


def create_visual_intro(shot_list, num):
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
    final_list = []
    i = 0
    for item in shot_list:
        img_path = item["shot_path"]
        label = item["label_text"]
        prefix = "[image" + str(num) + "_" + str(i) + "]"
        postfix = "is an example of " + label
        # if("caption" in item.keys()):
        #    postfix + item["caption"]
        item["visual_intro"] = [prefix, postfix]
        i += 1
        final_list.append(item)
    return final_list

def add_masked_process(dataset_path, new_dataset_path, mask_demo_flag):
    """
    向数据集中添加掩膜处理后的数据。

    Args:
        dataset_path (str): 原始数据集的路径。
        new_dataset_path (str): 处理后的数据集保存路径。
        mask_demo_flag (int): 是否添加掩膜处理的标志，大于0表示添加，否则不添加。

    Returns:
        None

    """

    total_list = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            if not "query_info" in data.keys():
                continue
            total_list.append(data)

    k = 0
    for item in total_list:
        k += 1
        if mask_demo_flag > 0:
            img_path = item["query_info"]["img_path"]
            for sparse_level in [1, 2, 3]:
                masked_path = (
                    img_path.replace(
                        "Manufacture", "Manufacture-mask-s" + str(sparse_level)
                    ).split(".")[0]
                    + ".png"
                )
                if os.path.exists(masked_path):
                    if not "masked_path" in item["query_info"].keys():
                        item["query_info"]["masked_path"] = {}
                    item["query_info"]["masked_path"][
                        "-mask-s" + str(sparse_level)
                    ] = masked_path

        save_log(new_dataset_path, item, mode="a")
    print(f"## Saving dataset ... {new_dataset_path}")
    return


def dataset_filter_process(
    log_ws,
    industry,
    dataset,
    success_dir,
    filter_log_path,
    dataset_path,
    score_threshold,
    least_model_num,
    sample_num,
):
    """
    根据成功路径过滤数据集并处理。

    Args:
        success_path (str): 成功路径的字符串表示，该路径指向包含JSON格式数据的文件。

    Returns:
        None

    该函数从成功路径指向的文件中读取每一行数据，解析JSON格式的数据，
    然后根据文件名中的实验编号和模型编号构建键（key），并将日志路径（log_path）作为值（value）
    存储到字典difficulty_filter_dict中。最后，打印出difficulty_filter_dict的键和对应的值。
    """

    difficulty_filter_dict = {}
    jsonl_list, _ = read_filename_from_dir(success_dir, "jsonl")
    dataset_list = []

    print(f"## Reading images from {jsonl_list}.")
    k = 0
    for success_path in jsonl_list:
        with open(success_path, "r", encoding="utf-8") as file:
            # print (success_path)
            for line in file:
                data = json.loads(line.strip())
                model = data["model"]
                key = "_".join([
                    str(data["benchmark_path"]),  
                    str(data["sample_num"]),
                    str(data["shot_num"]),
                    str(data["refine_prompt_flag"]),
                    str(data["demo_text_flag"]),
                    str(data["random_demo_flag"]),
                    str(data["mask_demo_flag"])
                ])
                key = "_".join(exp)
                if key in difficulty_filter_dict.keys():
                    difficulty_filter_dict[key][model] = data["log_path"]
                else:
                    difficulty_filter_dict[key] = {}
                    difficulty_filter_dict[key][model] = data["log_path"]

    print(f"## Considering {len(difficulty_filter_dict.keys())} exps in total.")

    dataset_dict = {}
    for key in difficulty_filter_dict.keys():
        exp_result = difficulty_filter_dict[key]
        for model in exp_result.keys():
            log_path = exp_result[model]
            if not log_ws in log_path:
                log_path = log_ws + log_path
            print(log_path)
            with open(log_path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line.strip())
                    if not "result_list" in data.keys():
                        continue
                    mark = data["result_list"][0][
                        -1
                    ]  # [timestamp, log_path, mode, response, mark]
                    img_path = data["query_info"]["img_path"]
                    if mark == "Error":
                        continue
                    if img_path in dataset_dict.keys():
                        dataset_dict[img_path]["scores"][model] = mark
                    else:
                        dataset_dict[img_path] = {"info": data, "scores": {}}
                        dataset_dict[img_path]["scores"][model] = mark

    filter_num = 0
    total_num = len(dataset_dict)
    candidate_img_list = []

    for key in dataset_dict:
        count = 0

        num = len(dataset_dict[key]["scores"].keys())
        for model in dataset_dict[key]["scores"]:
            count = count + dataset_dict[key]["scores"][model]
        score = count / num
        dataset_dict[key]["score"] = score

        if score < score_threshold and num > least_model_num:
            candidate_img_list.append(key)
            filter_num += 1

    print(
        f"## Finish difficulty filter with {filter_num} images, from {total_num} in total"
    )

    if sample_num > 0:
        # sample_list = random.sample(candidate_img_list, sample_num)
        sample_list = sample_by_diversity(candidate_img_list, sample_num)
    else:
        sample_list = candidate_img_list

    save_dataset_list = [dataset_dict[item]["info"] for item in sample_list]

    k += 1
    for item in save_dataset_list:
        item["data_info"]["id"] = str(k) + "_" + str(datetime.now())
        save_log(dataset_path, item, mode="a")


    print(f"## Saving dataset ... {dataset_path}")

    for key in sample_list:
        result = {
            "img_path": key,
            "score": dataset_dict[key]["score"],
            "score_detail": dataset_dict[key]["scores"],
            "info": dataset_dict[key]["info"],
        }
        save_log(filter_log_path, result, mode="a")

    save_num = len(sample_list)
    source_info = {}
    for item in save_dataset_list:
        source = item["data_info"]["source"]
        if source in source_info.keys():
            source_info[source] += 1
        else:
            source_info[source] = 1

    print(f"## Finish diversity filter with {save_num}, from {total_num} in total")
    save_log(
        filter_log_path,
        {
            "industry": industry,
            "dataset": dataset,
            "source_info": source_info,
            "score_threshold": score_threshold,
            "sample_num": sample_num,
            "success_dir": success_dir,
            "total_num": total_num,
            "filter_num": filter_num,
            "save_num": save_num,
        },
    )
    print(f"## Saving logs ... {filter_log_path}")


def dataset_add_fewshots_with_threads(
    thread_id,
    data_list,
    shot_target,
    shot_num,
    new_dataset_path,
    demo_text_flag,
    fine_prompt_flag,
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
    print(f"### no.{thread_id} thread running...")
    i = 0
    for data in data_list:
        prompt = data["query_info"]["instruct"]["clean"]
        img_path = data["query_info"]["img_path"]
        target = data["query_info"]["candidate"]
        shot_list = []
        demo_info = {}
        if shot_target == "each":
            img_list, _ = read_filename_from_dir(
                img_path.split(img_path.split("/")[-2])[0]
            )
            if not "demo_info" in data.keys():
                for label in target:
                    demo_info[label] = random_sample_fewshot(
                        img_list, img_path, label, shot_num, True
                    )
                    if demo_text_flag > 0:
                        demo_info[label] = create_detail_demonstration(
                            demo_info[label], target_num=-1, max_num=2
                        )
                    if demo_text_flag > 1:
                        demo_info[label] = create_visual_intro(demo_info[label], 2)
            else:
                demo_info = copy.deepcopy(data["demo_info"])
                for label in target:
                    for num in range(shot_num):
                        if (num > len(data["demo_info"][label]) - 1):
                            continue

                        if "caption" in data["demo_info"][label][num] and (
                            "error" in data["demo_info"][label][num]["caption"]
                            or "Error" in data["demo_info"][label][num]["caption"]
                        ):
                            print("found error in caption")
                            demo_info[label] = create_detail_demonstration(
                                data["demo_info"][label], target_num=num, max_num=10
                            )
        data["demo_info"] = demo_info

        if fine_prompt_flag > 0:
            new_prompt = (
                prompt.split("Constrain")[0]
                + "Constrain\n You should think step by step. "
                + "1. describe the image as detailed as possible. "
                + "2. think over the key issue you should take consider regrading to the task."
                + "3. give your reasoning step and answer the question. "
                + "4. write your answer following <answer> xxxx <answer> format."
            )
            data["query_info"]["instruct"]["cot"] = new_prompt
        if fine_prompt_flag > 1:
            if "detail" in data["query_info"]["instruct"]:
                if (
                    "error" in data["query_info"]["instruct"]["detail"]
                    or "Error" in data["query_info"]["instruct"]["detail"]
                ):
                    print("found error in detail")
                    new_prompt = refine_prompt(prompt)
                    if ("Error" in new_prompt or "error" in new_prompt):
                        print("[Warning] error in new_prompt", new_prompt)
                    data["query_info"]["instruct"]["detail"] = new_prompt
                    #print("new prompt is : ", new_prompt)
            else:
                new_prompt = refine_prompt(prompt)
                if ("Error" in new_prompt or "error" in new_prompt):
                    print("[Warning] error in new_prompt", new_prompt)
                data["query_info"]["instruct"]["detail"] = new_prompt

        save_log(new_dataset_path, data, "a")
        i = i + 1
        if i % 5 == 0:
            print(
                f"### {thread_id} Thread Processing {i} images, saving... to {new_dataset_path}"
            )

    return data_list


def dataset_add_fewshots(
    log_path,
    dataset_path,
    new_dataset_path,
    shot_num,
    shot_target,
    demo_text_flag=0,
    fine_prompt_flag=0,
):
    """
    为数据集添加少量样本

    Args:
        original_jsonl_path (str): 原始jsonl文件路径
        save_jsonl_path (str): 保存的jsonl文件路径
        shot_num (int): 要添加的样本数量
        few_target (str): 添加样本的目标字段

    Returns:
        None
    """
    i = 0
    total_list = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            if not "query_info" in data.keys():
                continue
            total_list.append(data)

    threads_num = 50
    if len(total_list) < threads_num:
        threads_num = len(total_list)

    print(f"## Beginning to start {threads_num} threads with {len(total_list)} data")

    chunk_size = len(total_list) // threads_num
    reminder = len(total_list) % threads_num

    threads = []
    for i in range(0, threads_num):  # 创建5个线程
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if i == threads_num - 1:
            end = end + reminder
        t = threading.Thread(
            target=dataset_add_fewshots_with_threads,
            args=(
                i,
                total_list[start:end],
                shot_target,
                shot_num,
                new_dataset_path,
                demo_text_flag,
                fine_prompt_flag,
            ),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()  # 等待所有线程完成

    print("## Saving dataset ...", new_dataset_path)
    return


def dataset_randomize(
    dataset_path,
    new_dataset_path,
    random_demo_flag,
):
    """
    随机打乱jsonl文件中数据的顺序。

    Args:
        original_jsonl_path (str): 输入的jsonl文件路径。
        save_jsonl_path (str): 保存打乱顺序后的jsonl文件的路径。
        random_demo_flag (int): 标志位，当值为1时执行打乱操作，否则不执行。

    Returns:
        None

    """
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            if not "demo_info" in data.keys():
                continue

            if random_demo_flag > 0:
                shot_dict = copy.deepcopy(data["demo_info"])
                true_label = data["query_info"]["label"]
                data["random_query_info"] = {}
                if type(data["query_info"]["candidate"]) == type("abc"):
                    candidate = json.loads(
                        data["query_info"]["candidate"].replace("'", '"')
                    )
                else:
                    candidate = data["query_info"]["candidate"]
                candidate_random = copy.deepcopy(candidate)
                random.SystemRandom().shuffle(candidate_random)
                candidate_dict = dict(zip(candidate, candidate_random))
                candidate_redict = dict(zip(candidate_random, candidate))

                for label in shot_dict.keys():
                    for i in range(0, len(shot_dict[label])):
                        #print(data["data_info"], shot_dict.keys(), data["demo_info"].keys())
                        shot_dict[label][i]["shot_path"] = data["demo_info"][
                            candidate_dict[label]
                        ][i]["shot_path"]
                data["random_query_info"]["random_label"] = candidate_redict[true_label]

                # for i in range(0, len(shot_dict[list(shot_dict.keys())[0]])):
                #     path_list = []
                #     for label in shot_dict.keys():
                #         path_list.append(shot_dict[label][i]["shot_path"])
                #     random.SystemRandom().shuffle(path_list)
                #     k = 0
                #     for label in shot_dict.keys():
                #         shot_dict[label][i]["shot_path"] = path_list[k]
                #         if true_label in path_list[k]:
                #             data["random_query_info"]["random_label"] = label
                #         k = k + 1
                data["random_demo_info"] = shot_dict

            if random_demo_flag > 1:
                new_shot_dict = {}
                new_candidate = []
                new_query_info = copy.deepcopy(data["query_info"])
                shot_dict = copy.deepcopy(data["demo_info"])
                for label in shot_dict.keys():
                    random_letters = [
                        random.choice(string.ascii_lowercase) for _ in range(5)
                    ]
                    new_label = "".join(random_letters)
                    new_candidate.append(new_label)
                    new_shot_dict[new_label] = []
                    for i in range(0, len(shot_dict[label])):
                        if new_label in new_shot_dict.keys():
                            new_shot_dict[new_label].append(
                                {
                                    "label_text": new_label,
                                    "shot_path": shot_dict[label][i]["shot_path"],
                                }
                            )
                        else:
                            new_shot_dict[new_label] = [
                                {
                                    "label_text": new_label,
                                    "shot_path": shot_dict[label][i]["shot_path"],
                                }
                            ]

                        if "caption" in shot_dict[label][i].keys():
                            new_shot_dict[new_label][i]["caption"] = shot_dict[label][
                                i
                            ]["caption"].replace(label, new_label)
                        if "visual_intro" in shot_dict[label][i].keys():
                            if "visual_intro" in new_shot_dict[new_label][i]:
                                new_shot_dict[new_label][i]["visual_intro"][
                                    0
                                ] = shot_dict[label][i]["visual_intro"][0].replace(
                                    label, new_label
                                )
                                new_shot_dict[new_label][i]["visual_intro"][
                                    -1
                                ] = shot_dict[label][i]["visual_intro"][-1].replace(
                                    label, new_label
                                )
                            else:
                                new_shot_dict[new_label][i]["visual_intro"] = [
                                    shot_dict[label][i]["visual_intro"][0].replace(
                                        label, new_label
                                    ),
                                    shot_dict[label][i]["visual_intro"][-1].replace(
                                        label, new_label
                                    ),
                                ]

                        if "clean" in new_query_info["instruct"].keys():
                            new_query_info["instruct"]["clean"] = new_query_info[
                                "instruct"
                            ]["clean"].replace(label, new_label)
                        if "detail" in new_query_info["instruct"].keys():
                            new_query_info["instruct"]["detail"] = new_query_info[
                                "instruct"
                            ]["detail"].replace(label, new_label)
                        if "cot" in new_query_info["instruct"].keys():
                            new_query_info["instruct"]["cot"] = new_query_info[
                                "instruct"
                            ]["cot"].replace(label, new_label)
                        if label == data["query_info"]["label"]:
                            new_query_info["label"] = new_label
                new_query_info["candidate"] = new_candidate
                data["fabricated_demo_info"] = new_shot_dict
                data["fabricated_query_info"] = new_query_info

            if random_demo_flag > 2:
                new_shot_dict = copy.deepcopy(data["demo_info"])
                for label in new_shot_dict.keys():
                    for i in range(0, len(new_shot_dict[label])):
                        new_shot_dict[label][i][
                            "shot_path"
                        ] = "src/images/white_image.png"
                data["blank_demo_info"] = new_shot_dict

            if random_demo_flag > 3:
                new_shot_dict = copy.deepcopy(data["demo_info"])
                for label in new_shot_dict.keys():
                    for i in range(0, len(new_shot_dict[label])):
                        j = random.randint(1, 10)
                        new_shot_dict[label][i]["shot_path"] = (
                            "src/images/noise_image_" + str(j) + ".png"
                        )
                data["noise_demo_info"] = new_shot_dict

            save_log(new_dataset_path, data, "a")
        print(f"## Saving dataset ... {new_dataset_path}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval the dataset")
    parser.add_argument("--log_ws", required=False, help="log_ws", default="")
    parser.add_argument("--success_dir", required=False, help="success_dir")
    parser.add_argument("--dataset", required=True, help="dataset")
    parser.add_argument("--industry", required=True, help="industry")
    parser.add_argument("--threshold", required=False, help="threshold", default=1.1)
    parser.add_argument(
        "--least_model_num", required=False, help="least_model_num", default=0
    )
    parser.add_argument("--sample_num", required=False, help="sample_num", default=10)

    parser.add_argument(
        "--dataset_path", required=False, help="dataset_path", default=""
    )
    parser.add_argument("--shot_target", required=False, help="shot_target")
    parser.add_argument("--shots_num", required=False, help="shots_num", default=0)
    parser.add_argument(
        "--demo_text_flag", required=False, help="demo_text_flag", default=0
    )
    parser.add_argument(
        "--fine_prompt_flag", required=False, help="fine_prompt_flag", default=0
    )
    parser.add_argument(
        "--random_demo_flag", required=False, help="random_demo_flag", default=0
    )
    parser.add_argument(
        "--mask_demo_flag", required=False, help="mask_demo_flag", default=0
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path

    info = (
        str(datetime.now().date())
        + "/"
        + args.industry
        + "_"
        + args.dataset
        + "_"
        + str(args.threshold)
        + "_"
        + str(args.least_model_num)
        + "_"
        + str(args.sample_num)
        + "_"
        + str(datetime.now().time())
    )
    if len(dataset_path) < 2:
        dataset_path = args.log_ws + "processed_data/stage2/dataset/" + info + ".jsonl"

    log_path = args.log_ws + "processed_data/stage2/log/" + info + ".jsonl"

    if len(args.success_dir) > 2 and len(args.dataset_path) < 2:
        print(
            f"# Beginning the process of diffculty filtering, "
            + f"threshold{args.threshold}, least_model_num{args.least_model_num}, sample_num{args.sample_num}"
        )

        dataset_filter_process(
            args.log_ws,
            args.industry,
            args.dataset,
            args.success_dir,
            log_path,
            dataset_path,
            float(args.threshold),
            int(args.least_model_num),
            int(args.sample_num),
        )
    else:
        print("# Skip the process of diffculty filtering")

    if int(args.shots_num) > 0 or int(args.fine_prompt_flag) > 0:
        print(
            f"# Beginning the process of fewshots and demo-text adding, with {args.shots_num} shots, "
            + f"demotextflag{args.demo_text_flag}, finepromptflag{args.fine_prompt_flag}"
        )
        new_dataset_path = (
            dataset_path.split(".jsonl")[0]
            + "_"
            + args.shot_target
            + "_"
            + str(args.shots_num)
            + "_shots_demotextflag"
            + str(args.demo_text_flag)
            + "_finepromptflag"
            + str(args.fine_prompt_flag)
            + ".jsonl"
        )
        save_log(new_dataset_path, {}, "w")
        dataset_add_fewshots(
            log_path,
            dataset_path,
            new_dataset_path,
            int(args.shots_num),
            args.shot_target,
            int(args.demo_text_flag),
            int(args.fine_prompt_flag),
        )
        dataset_path = new_dataset_path
    else:
        print("# Skip the process of fewshots adding")

    if int(args.random_demo_flag) > 0:
        print(
            f"# Beginning the process of random adding, randomdemoflag{args.random_demo_flag}"
        )
        new_dataset_path = (
            dataset_path.split(".jsonl")[0]
            + "_"
            + "randomedemoflag"
            + str(args.random_demo_flag)
            + ".jsonl"
        )
        save_log(new_dataset_path, {}, "w")
        dataset_randomize(
            dataset_path,
            new_dataset_path,
            int(args.random_demo_flag),
        )
        dataset_path = new_dataset_path
    else:
        print("# Skip the process of random adding")

    if int(args.mask_demo_flag) > 0:
        print(
            f"# Beginning the process of mased image adding, maskdemoflag {args.mask_demo_flag}"
        )
        new_dataset_path = (
            dataset_path.split(".jsonl")[0]
            + "_"
            + "maskdemoflag"
            + str(args.mask_demo_flag)
            + ".jsonl"
        )
        save_log(new_dataset_path, {}, "w")
        add_masked_process(
            dataset_path,
            new_dataset_path,
            int(args.mask_demo_flag),
        )
    else:
        print("# Skip the process of mask_demo_flag")

