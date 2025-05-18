# api_call.py
"""
this file is used to exp the llm's icl
"""
from utils.api_call import llm_model_api_gate
import csv
import random


def add_shot(prompt, shot_dict, random_demo_flag=0):
    """
    将示例添加到提示字符串中。

    Args:
        prompt (str): 要添加示例的提示字符串。
        shot_dict (dict): 包含示例的字典，键为标签，值为示例列表。
        random_demo_flag (int, optional): 是否随机选择示例的标志。默认为0，表示按顺序选择示例；设置为1表示随机选择示例。

    Returns:
        str: 包含示例的提示字符串。

    """
    if random_demo_flag == 1:
        key_list = list(shot_dict.keys())
        random.SystemRandom().shuffle(key_list)
    i = 0
    for key in shot_dict.keys():
        if random_demo_flag == 1:
            label = key_list[i]
            # print(f"true {key} and random as {label}\n")
        else:
            label = key
        prefix = (
            "Example of "
            + label
            + "\nInput: <begin>"
            + random.choice(shot_dict[key])
            + "<end>\n"
        )
        postfix = "Output: " + label + "\n"
        prompt = prompt + prefix + postfix
        i = i + 1
    return prompt


def main():
    """
    主函数，用于从CSV文件中读取诗歌情感数据，并通过GPT-4o模型进行情感分类。

    Args:
        无

    Returns:
        无

    """
    data_list = []
    shot_dict = {}
    with open(
        "/mnt/cfs_bj/liannan/visualicl_raw_data/LLM/final_df_emotions.csv",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row[0]) < 20:
                continue
            data_list.append(row)
            if row[2] in shot_dict.keys():
                shot_dict[row[2]].append(row[0])
            else:
                shot_dict[row[2]] = []

    for key in shot_dict:
        print(key, len(shot_dict[key]))

    class_list = list(shot_dict.keys())
    prompt = (
        "You are a poem emotion classifier, you will be given a sentence and the task is to predict its emotion.\n"
        + "The candidates emotions are [sadness, neutral, joy, fear, disgust, anger]\n"
        + "You should only output the direct answer without any explaination. The poem is as followed:\n"
    )

    right = 0
    total = 0
    error = 0
    for row in data_list:
        content = row[0]
        label = row[2]
        prompt = add_shot(prompt, shot_dict, 1)

        result = llm_model_api_gate(
            prompt + "\nInput: <begin>" + content + "<end>\n Output: ", "gpt-4.1"
        )

        if label.lower() == result.lower():
            right += 1

        if "error" in result or "Error" in result:
            error += 1
            print(f"error for request with {result}")
        else:
            total += 1

        print(f"---***--- right of {right} with total of {total} and error of {error}")

    print("acc:", right / total)


if __name__ == "__main__":
    main()
