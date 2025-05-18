# main.py
"""
This module contains the main function for the program.
"""
from utils.api_call import feature_description
from utils.api_call import create_img_from_feature
from utils.helper import download_image
from utils.helper import download_image_from_txt
from utils.api_call import mainland_vlm_api_call
from utils.helper import encode_image, save_log, check_success
from eval.run_evalset import inference_result_from_picdir
import json
import os
from datetime import datetime
import argparse
from eval.exp_setting import (
    exp_set_fromyaml,
    task_set_remoteSensing,
    task_set_tomatoDisease,
    task_set_DSMVTec,
    task_set_Pattern,
)


def eval_original_dataset(
    stage,
    log_ws,
    success_path,
    benchmark_path,
    test_vlm_model,
    level,
    sample_num=50,
    shot_num=0,
    refine_prompt_flag=0,
    demo_text_flag=0,
    random_demo_flag=0,
    mask_demo_flag=0,
    focus_dataset="all",
):
    """
    对原始数据集进行评估。

    Args:
        benchmark_dir (str): 结果文件的存储路径。
        test_vlm_model (str): 测试使用的视觉语言模型。
        sample_num (str): 样本数量。
        shot_num (str): shots的数量。
        shot_target (str): shots的目标。
        exp_flag (str): 实验标志。

    Returns:
        None

    """

    if not "jsonl" in benchmark_path:

        task_set = exp_set_fromyaml(
            "sh/exp_setting_for_level.yaml", benchmark_path, level
        )

        industry = task_set["industry"]
        scene = task_set["scene"]
        task = task_set["task"]
        level = task_set["level"]
        definition = task_set["definition"]
    else:
        dataset = benchmark_path.split("/")[-1].split("_")[1]
        industry = benchmark_path.split("/")[-1].split("_")[0]
        scene = dataset
        task = ""
        task_set = {}

    shot_num = int(shot_num)
    sample_num = int(sample_num)

    log_path = (
        log_ws
        + "log/"
        + stage
        + "/log_"
        + str(datetime.now().date())
        + "/"
        + str(datetime.now().time())
        + "_"
        + level
        + "_"
        + industry
        + "_"
        + scene
        + "_"
        + test_vlm_model
        + "_"
        + "focusdataset_"
        + focus_dataset
        + "_"
        + str(sample_num)
        + "samples_"
        + str(shot_num)
        + "shot_each"
        + "_refineprompt"
        + str(refine_prompt_flag)
        + "_demotext"
        + str(demo_text_flag)
        + "_randomdemoflag"
        + str(random_demo_flag)
        + "_maskdemoflag"
        + str(mask_demo_flag)
        + ".jsonl"
    )

    json_object = {
        "stage": stage,
        "log_path": log_path,
        "model": test_vlm_model,
        "focus_dataset": focus_dataset,
        "level": level,
        "industry": industry,
        "scene": scene,
        "task": task,
        "benchmark_path": benchmark_path,
        "sample_num": sample_num,
        "shot_num": shot_num,
        "refine_prompt_flag": refine_prompt_flag,
        "demo_text_flag": demo_text_flag,
        "random_demo_flag": random_demo_flag,
        "mask_demo_flag": mask_demo_flag,
    }

    succeed_dict = check_success(success_path, log_path)
    if len(succeed_dict.keys()) > 0:
        print(succeed_dict)
        num = succeed_dict["num"]
        if num > 1000 or not mask_demo_flag == 0:
            print("###Exp already done, skip")
            return
        else:
            log_path = succeed_dict["log_path"]
            print(
                f"## Exp Beginning as info: {json_object} from the {num} of {log_path}"
            )
    else:
        print(f"## Exp Beginning as info: {json_object} from new")
        save_log(log_path, json_object)

    json_object["exp_result"] = inference_result_from_picdir(
        benchmark_path,
        task_set,
        test_vlm_model,
        log_path,
        success_path,
        sample_num,
        shot_num,
        focus_dataset,
        refine_prompt_flag,
        demo_text_flag,
        random_demo_flag,
        mask_demo_flag,
    )
    good_num = json_object["exp_result"]["Total"][1]
    if good_num > 1:
        # vaild total num > 1 could be consdiered as success
        print(f"## Exp Succeeded, saving to {success_path}")
        save_log(success_path, json_object)
    else:
        print(f"## Exp not Succeeded, due to good log num {good_num} <= 1")


def dataset_generation_process():
    """
    用于处理和展示特征信息，并生成图像。

    Args:
        无

    Returns:
        无

    """
    dimension = "element"
    industry = "E-commence"
    scene = "Cutout Defect Detection"
    # industry = "remoteSensing"
    # scene = "photovoltaic panel identification in satellite imagery, especially from a very distant perspective"
    postive_num = 6
    negative_num = 4
    model = "deepseek-r1"  # "gemini-2.0-flash-exp"
    result = feature_description(
        dimension, industry, scene, postive_num, negative_num, model
    )
    if "think" in result:
        result = "{" + result.split("{")[-1]
        result = result.split("}")[0] + "}"
    # print(result)
    feature_dict = json.loads(result)

    # name = "positive_hard_1"
    llm_model = "deepseek-r1"
    sd_model = "gemini-2.0-flash-exp"

    experment_name = "exp" + industry + "_" + scene + "_" + str(datetime.now())
    result_dir = "test/" + experment_name + "/"
    log_path = "test/" + experment_name + ".txt"
    os.makedirs(result_dir, exist_ok=True)

    for name in feature_dict.keys():
        if not "hard" in name:
            continue
        feature = feature_dict[name]
        print("## Focus feature\n", feature)
        for i in range(0, 3):
            url = create_img_from_feature(
                dimension, industry, scene, name, feature, llm_model, sd_model
            )
            print("## Final Url", url)
            url = "http" + url.split("http")[-1]
            save_path = result_dir + name + "_num" + str(i) + "_0328.jpg"
            download_image(url, save_path, log_path)
    download_image_from_txt(log_path)
    # eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval the dataset")
    parser.add_argument("--stage", required=True, help="the stage of the experiments")
    parser.add_argument(
        "--success_path", required=True, help="path to store successful experiments"
    )
    parser.add_argument(
        "--benchmark_dir", required=False, help="benchmark directory or jsonl"
    )
    parser.add_argument("--log_ws", required=False, help="log_ws", default="")
    parser.add_argument(
        "--vlm",
        required=True,
        help="eval vlm model",
        choices=[
            "random-policy",
            "doubao-1.5-vision-pro-32k-250115",
            "Qwen*QVQ-72B-Preview",
            "Qwen*Qwen2.5-VL-72B-Instruct",
            "deepseek-ai*deepseek-vl2",
            "gemini-2.5-pro-exp-03-25",
            "gpt-4o",
            "gpt-4.1",
            "<gpt>o3",
            "<gpt>o4-mini",
            "think<gpt>o3",
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-20250219-thinking",
            "ernie-4.5-8k-preview",
            "ernie-4.5-turbo-vl-32k",
            "internvl-78b",
        ],
    )
    parser.add_argument(
        "--sample_num",
        required=False,
        help="number of samples to evaluate on",
        default=500,
    )
    parser.add_argument(
        "--shot_num", required=False, help="number of few-shot examples", default=0
    )
    parser.add_argument(
        "--refine_prompt_flag",
        required=False,
        help="refine prompt or not",
        default=0,
    )
    parser.add_argument(
        "--demo_text_flag",
        required=False,
        help="caption the shots or not",
        default=0,
    )
    parser.add_argument(
        "--level",
        required=False,
        help="obsearvtion level",
        choices=["detail", "pattern", "element"],
        default="detail",
    )
    parser.add_argument(
        "--focus_dataset",
        required=False,
        help="obsearvtion focus_dataset",
        default="all",
    )
    parser.add_argument(
        "--random_demo_flag",
        required=False,
        help="random_demo_flag",
        default=0,
    )
    parser.add_argument(
        "--mask_demo_flag",
        required=False,
        help="mask_demo_flag",
        default=0,
    )

    args = parser.parse_args()
    eval_original_dataset(
        args.stage,
        args.log_ws,
        args.success_path,
        args.benchmark_dir,
        args.vlm,
        args.level,
        int(args.sample_num),
        int(args.shot_num),
        int(args.refine_prompt_flag),
        int(args.demo_text_flag),
        int(args.random_demo_flag),
        int(args.mask_demo_flag),
        args.focus_dataset,
    )
