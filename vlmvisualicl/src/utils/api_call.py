# api_call.py
"""
this file is used to call the api of the model
"""

import requests
import json
import sys
import os
import random
from openai import OpenAI
from utils.helper import encode_image, prompt_assemble, read_filename_from_dir
from utils.helper import save_log, cal_metric, random_sample_fewshot
import ast


def create_detail_caption(shot_list):
    """
    根据提供的shot_list生成带有详细描述的列表。

    Args:
        shot_list (list): 包含图像路径和标签文本的列表，每个元素都是一个字典，包含两个键：'shot_path'和'label_text'。

    Returns:
        list: 返回一个列表，每个元素都是一个包含原始图像路径、标签文本和详细描述的字典。

    """
    final_list = []
    for item in shot_list:
        img_path = item["shot_path"]
        text = item["label_text"]

        url = encode_image(img_path)

        instruction = (
            "You are an image detail observer."
            + "You should give detailed demonstration of the image as much as possible,"
            + " especially considering the industry defeat recognition task."
            + "\n Please just give the detailed demonstration with one paragraph and give your answer in Chinese."
        )
        model = "doubao-1.5-vision-pro-32k-250115"
        result = vlm_model_api_gate(instruction, [], url, model)
        if "Error" in result or "error" in result:
            print(f"[Error]: API request failed in create_detail_caption", result)
            item["caption"] = ""
        else:
            item["caption"] = result

        # print("***", demonstration)
        final_list.append(item)
    return final_list


def create_detail_demonstration(shot_list, target_num=-1, max_num=10):
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
    
    model = "gpt-4.1"  # "doubao-1.5-vision-pro-32k-250115"
    if(target_num > -1):
        item = shot_list[target_num]
        img_path = item["shot_path"]
        text = item["label_text"]
        url = encode_image(img_path)
        instruction = (
            "You are an image detail observer."
            + "You should give detailed demonstration of the image to prove it could be classified as "
            + text
            + "\n Please just give the detailed demonstration with one paragraph."
        )
        shot_list[target_num]["caption"] = vlm_model_api_gate(instruction, [], url, model)
        return shot_list

    i = 0
    final_list = []
    for item in shot_list:
        if i < max_num:
            img_path = item["shot_path"]
            text = item["label_text"]
            url = encode_image(img_path)
            instruction = (
                "You are an image detail observer."
                + "You should give detailed demonstration of the image to prove it could be classified as "
                + text
                + "\n Please just give the detailed demonstration with one paragraph."
            )
            item["caption"] = vlm_model_api_gate(instruction, [], url, model)
        # print("***", demonstration)
        final_list.append(item)
        i += 1
    return final_list


def refine_prompt(prompt):
    """
    根据原始提示生成更详细的提示。

    Args:
        prompt (str): 原始提示。

    Returns:
        str: 基于原始提示生成的更详细的提示。

    """
    object_list = ast.literal_eval("[" + prompt.split("[")[-1].split("]")[0] + "]")
    instruction = (
        "You are a prompt generator. You should generate new prompt based on the original one "
        + "with detailed knowledge and information to help the user finish a recogniztion task. "
        + "You should give enough descrition about each the category and you should not change the specific candidante name\n The original prompt is: \n </start>\n "
        + prompt
        + "\n</end>\n"
    )
    model = "deepseek-ai*DeepSeek-R1"
    count = 0
    while True:
        if count > 3:
            return ""
        response = llm_model_api_gate(instruction, model)
        result = response.split("</think>")[-1]
        for item in object_list:
            if not item in result:
                count += 1
                continue
        break
    return result


def create_img_from_feature(
    dimension, industry, scene, name, feature, llm_model, sd_model
):
    """
    根据特征创建图像URL。

    Args:
        dimension (str): 图像的尺寸，例如 "512x512"。
        industry (str): 行业类型，例如 "艺术" 或 "科技"。
        scene (str): 场景类型，例如 "室内" 或 "室外"。
        name (str): 图像的名称。
        feature (str): 图像的特征描述，例如 "美丽的日落"。
        llm_model (str): 用于生成提示的LLM模型名称。
        sd_model (str): 用于生成图像的SD模型名称。

    Returns:
        str: 生成的图像URL。

    """
    prompt = create_prompt_from_feature(
        dimension, industry, scene, name, feature, llm_model
    )
    if "think" in prompt:
        prompt = prompt.split("</think>")[-1]

    prompt = prompt + "\n##Constrain\n Just give me the img and no other explain"

    if "dall" in sd_model:
        url = oversea_dalle_api_call(prompt, sd_model)
    else:
        url = oversea_sd_api_call(prompt, sd_model)

    return url


def create_prompt_from_feature(dimension, industry, scene, name, feature, model):
    """
    根据给定的特征创建提示信息。

    Args:
        dimension (str): 维度，如"宏观"或"微观"。
        industry (str): 行业名称，如"金融"或"医疗"。
        scene (str): 场景名称，如"报告"或"广告"。
        name (str): 对象名称，如"建筑"或"产品"。
        feature (str): 特征描述，如"现代感"或"科技感"。
        model (str): 使用的模型名称。

    Returns:
        str: 根据输入特征生成的提示信息。

    """
    definition = "independently identifiable constituent units in an object, including physical components or abstract elements. The element is usually occupying very small part of the image"
    prompt_template = "## Role\n You are an expert in {{industry}} industry and ready for drawing a image material for the task of {{scene}}. You should take care of {{dimension}} level information, noted as {{definition}}. \n ## Task \n You task is giving a prompt to drawing a {{name}} sample, while taking consideration of  {{feature}} \n##Constrain\n Just give me the prompt and no other explain"

    prompt = prompt_assemble(prompt_template, "industry", industry)
    prompt = prompt_assemble(prompt, "scene", scene)
    prompt = prompt_assemble(prompt, "dimension", dimension)
    prompt = prompt_assemble(prompt, "definition", definition)
    prompt = prompt_assemble(prompt, "name", name)
    prompt = prompt_assemble(prompt, "feature", feature)
    return oversea_llm_api_call(prompt, model)


def feature_description(dimension, industry, scene, positive_num, negative_num, model):
    """
    生成特定行业、场景、维度下的正负样本特征描述。

    Args:
        dimension (str): 特征的维度描述。
        industry (str): 行业名称。
        scene (str): 场景名称。
        positive_num (int): 正样本数量。
        negative_num (int): 负样本数量。
        model (str): 使用的模型名称。

    Returns:
        str: 生成的特征描述。

    """

    prompt_template = "## Role\n You are an expert in {{industry}} industry and investigate the job of {{scene}}. You should search for some information and consider the potential difficulty in this task while taking care of {{dimension}} level promblem, noted as {{definition}}. \n ## Task \n You task is listing {{positive_num}} aspect of positive sample, including half easy and half hard ones, and then the give {{negative_num}} direction of negative sample,including half easy and half hard ones. \n ## Constrain \n1. Just give your final answer in English. 2. List in Json Format {'positive_easy_1':'xxxx','positive_hard_1':'xxxx','negative_easy_1':'xxxx','negative_hard_1':'xxxx'}. 3. Just the json object and no more other explain"
    definition = "independently identifiable constituent units in an object, including physical components or abstract elements. The element is usually occupying very small part of the image"

    prompt = prompt_assemble(prompt_template, "industry", industry)
    prompt = prompt_assemble(prompt, "scene", scene)
    prompt = prompt_assemble(prompt, "dimension", dimension)
    prompt = prompt_assemble(prompt, "definition", definition)
    prompt = prompt_assemble(prompt, "positive_num", str(positive_num))
    prompt = prompt_assemble(prompt, "negative_num", str(negative_num))

    return oversea_llm_api_call(prompt, model)


def post_model_api_call(prompt, shot_list, url, model):
    """
    调用Post API进行文本生成。

    Args:
        prompt (str): 用户输入的问题或提示。
        shot_list (list): 示例列表，用于指导模型生成类似文本。
        url (str): 千帆LLM API的URL地址。
        model (str): 使用的模型名称。

    Returns:
        str: API返回的JSON格式的文本内容。

    """
    # print("model:", model)
    if "doubao" in model:
        service_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        # key = "***"
        key = "***"  # personal key
        model = "doubao-vision-pro-32k-241028"
    elif "deepseek" in model:
        service_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        key = "***"  # personal key
        model = "deepseek-r1-250120"
    elif "ernie" in model:
        service_url = "https://qianfan.baidubce.com/v2/chat/completions"
        key = "***"
    elif "intern" in model:
        service_url = "https://api.friendli.ai/dedicated/v1/chat/completions"
        key = "***"
        model = "2c97kji724fc"  # internvl-78b

    content = [{"type": "text", "text": prompt}]
    for shot in shot_list:
        content.append({"type": "image_url", "image_url": {"url": shot[1]}})
        content.append({"type": "text", "text": shot[2]})
    if len(url) > 0:
        content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        print("no url here!")
    response = ""
    try:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": content}],
            }
        )
        headers = {
            "Content-Type": "application/json",
            "appid": "",
            "Authorization": "Bearer " + key,
        }

        response = requests.request(
            "POST", service_url, headers=headers, data=payload.encode("utf-8")
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        if type(response) == type("1"):
            return f"Error in reference, {type(e).__name__}, {str(e)}, response: {response}"
        else:
            return f"Error in reference, {type(e).__name__}, {str(e)}, response: {response.json()}"


def mainland_vlm_api_call(prompt, shot_list, url, model, retry=0):
    """
    调用VLM API进行文本生成。

    Args:
        prompt (str): 用户输入的问题或提示。
        shot_list (list): 示例列表，用于指导模型生成类似文本。
        url (str): 千帆VLM API的URL地址。
        model (str): 使用的模型名称。
        retry (int, optional): 重试次数，默认为0。

    Returns:
        str: API返回的JSON格式的文本内容。

    """
    api_key_list = [
        "***",
        "***",
        "***",
        "***",
        "***",
        "***",
        "***",
        "***",
        "***",
        "***",
    ]
    retry = retry % len(api_key_list)
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        # api_key="***", # personal key
        api_key=api_key_list[retry],
    )
    content = [{"type": "text", "text": prompt}]
    for shot in shot_list:
        content.append({"type": "image_url", "image_url": {"url": shot[1]}})
        content.append({"type": "text", "text": shot[2]})
    if len(url) > 0:
        content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        print("no url here!")
    response = ""
    try:
        response = client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model=model,  # "doubao-1-5-vision-pro-32k-250115",
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        # print(result)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in reference, {type(e).__name__}, {str(e)}, response: {response}"


def llm_model_api_gate(prompt, model, retry=5, thread_id=0):
    """
    根据提供的提示、镜头列表、URL和模型名称调用相应的API并返回结果。

    Args:
        prompt (str): 提示信息。
        shot_list (list): 镜头列表。
        url (str): 请求的URL。
        model (str): 模型名称。
        retry (int, optional): 重试次数，默认为2。

    Returns:
        str: API调用的结果。

    """
    for i in range(0, retry):
        # result = post_model_api_call(prompt, [], "", model)
        if "deepseek" in model:
            result = siliconflow_vlm_api_call(prompt, [], "", model, thread_id + 6 * i)
        else:
            result = oversea_llm_api_call(prompt, model)
        if "Error" in result or "error" in result:
            continue
        else:
            return result
    return result


def vlm_model_api_gate(prompt, shot_list, url, model, retry=2, thread_id=0):
    """
    根据提供的提示、镜头列表、URL和模型名称调用相应的API并返回结果。

    Args:
        prompt (str): 提示信息。
        shot_list (list): 镜头列表。
        url (str): 请求的URL。
        model (str): 模型名称。
        retry (int, optional): 重试次数，默认为2。

    Returns:
        str: API调用的结果。

    """
    for i in range(0, retry):
        # result = post_model_api_call(prompt, shot_list, url, model)
        if "doubao" in model:
            result = mainland_vlm_api_call(
                prompt, shot_list, url, model, thread_id + 6 * i
            )
        elif "Qwen" in model or "deepseek" in model:
            result = siliconflow_vlm_api_call(
                prompt, shot_list, url, model, thread_id + 6 * i
            )
        elif "intern" in model or "ernie" in model:
            result = post_model_api_call(prompt, shot_list, url, model)
        else:
            if "<gpt>" in model:
                model = model.split("<gpt>")[-1]
            result = oversea_vlm_api_call(prompt, shot_list, url, model)

        if "Error" in result or "error" in result:
            continue
        else:
            return result
    return result


def siliconflow_vlm_api_call(prompt, shot_list, url, model, retry=0):
    """
    向siliconflow VLM API发送请求，并返回响应结果。

    Args:
        prompt (str): 提示文本。
        shot_list (list of tuple): 包含图片URL和对应说明的元组列表，格式如[(说明1, 图片URL1), (说明2, 图片URL2), ...]。
        url (str): 图片URL。
        model (str): 使用的模型名称。

    Returns:
        dict or str: 如果请求成功，返回API返回的JSON响应结果；如果请求失败，返回错误消息。

    Raises:
        requests.exceptions.RequestException: 如果请求过程中出现网络错误。
        KeyError: 如果响应JSON中缺少必要的键。
        json.JSONDecodeError: 如果响应不是有效的JSON格式。

    """
    if model == "Qwen*Qwen2.5-VL-72B-Instruct":
        model = "qwen2.5-vl-72b-instruct"
    elif model == "Qwen*QVQ-72B-Preview":
        model = "qvq-72b-preview"
    elif model == "deepseek-ai*deepseek-vl2":
        model = "deepseek-vl2"
    else:
        return f"Error in reference, model name not found: {model}"

    # print("## API CALL ##\n Asking...",model,"\n",prompt,"\n")
    #API_URL = "https://api.siliconflow.cn/v1/chat/completions"
    API_URL = "http://110.42.43.30:3000/v1/chat/completions"
    api_key_list = [
        "***"
    ]
    retry = retry % len(api_key_list)
    API_KEY = api_key_list[retry]

    headers = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

    content = [{"type": "text", "text": prompt}]
    for shot in shot_list:
        content.append({"type": "image_url", "image_url": {"url": shot[1]}})
        content.append({"type": "text", "text": shot[2]})
    if len(url) > 1:
        content.append({"type": "image_url", "image_url": {"url": url}})

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    response = ""

    try:
        response = requests.request("POST", API_URL, json=data, headers=headers)
        # response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        # response.raise_for_status()  # 如果请求失败则抛出异常

        result = response.json()["choices"][0]["message"]["content"]
        if "Final Answer" in result:
            result = result.split("Final Answer")[-1]
        # print(result)
        return result
    except Exception as e:
        if type(response) == type("1"):
            return f"Error in reference, {type(e).__name__}, {str(e)}, response: {response}"
        else:
            return f"Error in reference, {type(e).__name__}, {str(e)}, response: {response.json()}"


def oversea_vlm_api_call(prompt, shot_list, url, model):
    """
    向海外VLM API发送请求，并返回响应结果。

    Args:
        prompt (str): 提示文本。
        shot_list (list of tuple): 包含图片URL和对应说明的元组列表，格式如[(说明1, 图片URL1), (说明2, 图片URL2), ...]。
        url (str): 图片URL。
        model (str): 使用的模型名称。

    Returns:
        dict or str: 如果请求成功，返回API返回的JSON响应结果；如果请求失败，返回错误消息。

    Raises:
        requests.exceptions.RequestException: 如果请求过程中出现网络错误。
        KeyError: 如果响应JSON中缺少必要的键。
        json.JSONDecodeError: 如果响应不是有效的JSON格式。

    """
    #print("## API CALL ##\n Asking...",model,"\n",prompt,"\n")
    #API_URL = "http://ai.wenyue8.com:15588/v1/chat/completions"
    #API_URL = "http://211.23.3.237:15588/v1/chat/completions"
    API_URL = "http://211.23.3.237:15588/v1/chat/completions"
    API_KEY = "***"

    headers = {"Content-Type": "application/json", "Authorization": f"{API_KEY}"}
    content = [{"type": "text", "text": prompt}]
    for shot in shot_list:
        content.append({"type": "image_url", "image_url": {"url": shot[1]}})
        content.append({"type": "text", "text": shot[2]})
    if len(url) > 1:
        content.append({"type": "image_url", "image_url": {"url": url}})

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    response = ""
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # 如果请求失败则抛出异常

        result = response.json()
        #print("\n -- Answer: ", result)
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error in reference, {type(e).__name__}, {str(e)}, response: {response}"


def oversea_dalle_api_call(prompt, model):
    """
    调用海外DALLE API生成图像。

    Args:
        prompt (str): 生成图像的提示文本。
        model (str): 使用的模型名称。

    Returns:
        str: 返回生成的图像URL。如果请求或解析失败，则返回错误信息。

    Raises:
        None

    """
    print("## API CALL ##\n Asking...", model, "\n", prompt, "\n")
    API_URL = "http://ai.wenyue8.com:15588/v1/images/generations"
    API_KEY = "***"

    prompt = (
        prompt
        + "The target element is usually occupying very small part of the image. Real world style"
    )
    headers = {"Content-Type": "application/json", "Authorization": f"{API_KEY}"}

    data = {"model": model, "prompt": prompt, "size": "1024x1024", "n": 1}

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # 如果请求失败则抛出异常

        result = response.json()
        # print(result)
        return result["data"][0]["url"]
    except requests.exceptions.RequestException as e:
        return f"请求错误: {e}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"解析响应错误: {e}\n原始响应: {response.text}"


def oversea_sd_api_call(prompt, model):
    """
    调用海外模型生成文本。

    Args:
        prompt (str): 用户输入的文本。
        model (str): 使用的模型名称。

    Returns:
        str: 生成的文本。

    Raises:
        None

    """
    print("## API CALL ##\n Asking...", model, "\n", prompt, "\n")
    API_URL = "http://ai.wenyue8.com:15588/v1/chat/completions"
    API_KEY = "***"

    prompt = (
        prompt + "The target element is usually occupying very small part of the image"
    )
    headers = {"Content-Type": "application/json", "Authorization": f"{API_KEY}"}

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # 如果请求失败则抛出异常

        result = response.json()
        # print(result)
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"请求错误: {e}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"解析响应错误: {e}\n原始响应: {response.text}"


def oversea_llm_api_call(prompt, model):
    """
    调用海外LLM API进行文本生成。

    Args:
        prompt (str): 用户输入的文本提示。
        model (str): 要使用的LLM模型名称。

    Returns:
        str: API返回的生成文本。

    Raises:
        requests.exceptions.RequestException: 请求过程中发生异常时抛出。
        KeyError, json.JSONDecodeError: 解析API响应时发生异常时抛出。

    """
    # print("## API CALL ##\n Asking...", model, "\n", prompt, "\n")
    API_URL = "http://ai.wenyue8.com:15588/v1/chat/completions"
    API_KEY = "***"

    headers = {"Content-Type": "application/json", "Authorization": f"{API_KEY}"}

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # 如果请求失败则抛出异常

        result = response.json()
        # print(result)
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error in reference, {type(e).__name__}, {str(e)}, response: {response.text}"
