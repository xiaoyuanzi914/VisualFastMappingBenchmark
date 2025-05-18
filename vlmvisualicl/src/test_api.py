# run_evalset.py
"""
This file contains the code for testing the model api.
"""
from utils.api_call import (
    mainland_vlm_api_call,
    oversea_vlm_api_call,
    siliconflow_vlm_api_call,
)
from utils.api_call import post_model_api_call, vlm_model_api_gate, llm_model_api_gate
from utils.helper import encode_image
import base64
import requests
import threading


def run_for_once(model):
    """
    执行一次模型运行。

    Args:
        model (str): 要运行的模型名称。

    Returns:
        None

    """
    url = encode_image("src/images/white_image.png")
    shot_info = []
    prompt = "What's in the img?"

    print("start...", model)
    for i in range(1):
        # result = post_model_api_call(prompt, shot_info, url, model)
        result = vlm_model_api_gate(prompt, shot_info, url, model)
        #result = llm_model_api_gate(prompt, model)
        # if "doubao" in model:
        #     result = mainland_vlm_api_call(prompt, shot_info, url, model)
        # elif "Qwen" in model:
        #     result = siliconflow_vlm_api_call(prompt, shot_info, url, model, i)
        # elif "ernie" in model or "intern" in model:
        #     result = post_model_api_call(prompt, shot_info, url, model)
        # else:
        #     result = oversea_vlm_api_call(prompt, shot_info, url, model)
        print(i, result)
    return


if __name__ == "__main__":

    run_for_once("gemini-2.5-pro-exp-03-25")
    # threads = []
    # for i in range(5):  # 创建5个线程
    #     t = threading.Thread(target=run_for_once, args=("gpt-4.1",))
    #     threads.append(t)
    #     t.start()

    # for t in threads:
    #     t.join()  # 等待所有线程完成

    # model = "ernie-3.5-8k"
    # run_for_once(model)

    # "ernie-3.5-8k"
    # "gemini-2.5-pro-exp-03-25",
    # "gemini-2.5-flash-preview-04-17"
    # "doubao-1.5-vision-pro-32k-250115",
    # "doubao-1.5-vision-pro-32k-250328",
    # "gpt-4o",
    # "Qwen*QVQ-72B-Preview",
    # "Qwen*Qwen2.5-VL-72B-Instruct",
