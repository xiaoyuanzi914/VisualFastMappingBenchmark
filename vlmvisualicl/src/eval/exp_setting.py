# exp_setting.py
"""
This file contains the code for exp_setting.
"""

import yaml


def exp_set_fromyaml(yaml_path, benchmark_dir, target_level):
    """
    从yaml文件中读取配置，生成任务集。

    Args:
        yaml (str): yaml文件的路径。
        benchmark_dir (str): 基准测试目录的路径。
        target_level (str): yaml文件中对应的层级。

    Returns:
        dict: 包含任务集信息的字典。

    """
    benchmark_dir = benchmark_dir.split("raw_data/")[-1]
    print(benchmark_dir)
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if(not target_level in config.keys()):
        print(f"[Error] target_level not found for {target_level}, please check {yaml_path}!")
        return -1
    config = config[target_level]
    scene_postfix = config["scene_postfix"]
    task = config["task"]
    cate = config["cate"]
    level = config["level"]
    definition = config["definition"]
        
    industry = benchmark_dir.split("/")[0]
    dataset = benchmark_dir.split("/")[1]
    scene = benchmark_dir.split("/")[2]
    scene = scene + scene_postfix

    task_set = {
        "industry": industry,
        "scene": scene,
        "cate": cate,
        "task": task,
        "level": level,
        "definition": definition,
    }
    return task_set


def task_set_Pattern(benchmark_dir):
    """
    定义任务集合，用于设置拉索瑕疵识别任务。

    Args:
        无

    Returns:
        dict: 包含行业、场景、任务、类别、维度和定义的任务集合。

    """
    industry = benchmark_dir.split("/")[0]
    dataset = benchmark_dir.split("/")[1]
    scene = benchmark_dir.split("/")[2]
    scene = scene + "Recogization"
    task = "obsearving the image and determine whether it could be classified as one of the candidate categorie "
    cate = [" "]  # focus class, useful for binary exp
    level = "Pattern"
    definition = (
        "identifying and localizing objects that exhibit specific structural, "
        + "repetitive, or contextual patterns"
    )
    task_set = {
        "industry": industry,
        "scene": scene,
        "task": task,
        "cate": cate,
        "level": level,
        "definition": definition,
    }
    return task_set


def task_set_DSMVTec(benchmark_dir):
    """
    定义任务集合，用于设置拉索瑕疵识别任务。

    Args:
        无

    Returns:
        dict: 包含行业、场景、任务、类别、维度和定义的任务集合。

    """
    industry = benchmark_dir.split("/")[0]
    dataset = benchmark_dir.split("/")[1]
    scene = benchmark_dir.split("/")[2]
    scene = scene + "DefeatIdentification"
    task = "obsearving the image and determine whether it could be classified as one of the candidate categorie "
    cate = [" "]  # focus class, useful for binary exp
    level = "Detail"
    definition = " Fine-grained visual features appearing on the surface of objects that are independent"
    task_set = {
        "industry": industry,
        "scene": scene,
        "task": task,
        "cate": cate,
        "level": level,
        "definition": definition,
    }
    return task_set


def task_set_remoteSensing():
    """
    设置遥感任务

    Args:
        无

    Returns:
        dict: 包含遥感任务相关信息的字典，具体包含以下字段：
            - industry (str): 行业名称，值为 "RemoteSensing"
            - scene (str): 场景名称，值为 "SatelliteImageIdentification"
            - task (str): 任务描述，值为 "观察图像并确定其是否可以归类为候选类别之一"
            - cate (list): 类别列表，值为 ["Pasture"]，表示关注类别，适用于二元实验
            - level (str): 维度描述，值为 "Pattern"
            - definition (str): 定义描述，值为 "由重复单元或复杂拓扑结构组成的介观视觉规律"

    """
    industry = "RemoteSensing"
    scene = "SatelliteImageIdentification"
    task = "obsearving the image and determine whether it could be classified as one of the candidate categorie "
    cate = ["Pasture"]  # focus class, useful for binary exp
    level = "Pattern"
    definition = " mesoscopic visual regularities consisting of repeating units or complex topologies "
    task_set = {
        "industry": industry,
        "scene": scene,
        "task": task,
        "cate": cate,
        "level": level,
        "definition": definition,
    }
    return task_set


def task_set_tomatoDisease():
    """
    定义任务集合，用于设置番茄病害识别任务。

    Args:
        无

    Returns:
        dict: 包含行业、场景、任务、类别、维度和定义的任务集合。

    """
    industry = "Agriculture"
    scene = "TomatoDiseaseIdentification"
    task = "obsearving the image and determine whether it could be classified as one of the candidate categorie "
    cate = ["LeafMold"]  # focus class, useful for binary exp
    level = "Detail"
    definition = " Fine-grained visual features appearing on the surface of objects that are independent"
    task_set = {
        "industry": industry,
        "scene": scene,
        "task": task,
        "cate": cate,
        "level": level,
        "definition": definition,
    }
    return task_set
