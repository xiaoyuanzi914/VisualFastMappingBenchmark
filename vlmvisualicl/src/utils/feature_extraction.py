
# feature_extraction.py
"""
This module contains the function for feature extraction
"""
import torch
import base64
import requests
import json
import sys
import numpy as np
from sklearn.cluster import KMeans

class FeatureExtractor:
    def __init__(self, text_image_service_url="http://10.93.150.29:8854", doc_service_url=None):
        """
        初始化特征提取器客户端
        
        Args:
            text_image_service_url: 文本和图像特征服务的基础URL
            doc_service_url: 文档图像特征服务的基础URL，如果为None则使用text_image_service_url
        """
        self.text_image_service_url = text_image_service_url
        self.doc_service_url = doc_service_url if doc_service_url else text_image_service_url
        self.headers = {'Content-Type': 'application/json'}
        
    def extract_text_features(self, texts, context_length=52):
        """
        提取文本特征，通过调用feature_service服务
        
        Args:
            texts: 文本列表
            context_length: 上下文长度
            
        Returns:
            文本特征向量
        """
        # 准备请求数据
        payload = {
            "texts": texts,
            "context_length": context_length
        }
        
        # 发送请求到文本特征提取服务
        response = requests.post(
            f"{self.text_image_service_url}/extract_text_features", 
            headers=self.headers, 
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "features" in response_data:
                # 将特征转换为tensor
                features = torch.tensor(response_data["features"])
                return features
            else:
                raise Exception(f"服务返回的数据中没有特征: {response_data}")
        else:
            raise Exception(f"服务请求失败，状态码: {response.status_code}, 响应: {response.text}")
    
    def extract_image_features(self, image_path):
        """
        提取图像特征，通过调用feature_service服务
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像特征向量
        """
        # 读取并编码图像
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # 准备请求数据
        payload = {
            "image_base64": image_base64
        }
        
        # 发送请求到图像特征提取服务
        response = requests.post(
            f"{self.text_image_service_url}/extract_image_features", 
            headers=self.headers, 
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "features" in response_data:
                # 将特征转换为tensor
                features = torch.tensor(response_data["features"])
                return features
            else:
                raise Exception(f"No features in the data: {response_data}")
        else:
            raise Exception(f"Fail to request，status code: {response.status_code}, response: {response.text}")
    

def feature_extract(image_path):
    """
    从给定路径的图像中提取特征。

    Args:
        image_path (str): 图像文件的路径。

    Returns:
        list: 提取的特征列表。

    Raises:
        Exception: 如果在提取特征过程中发生错误，则抛出异常。

    """
    service_url = "http://10.93.150.29:8854"
    try:
        # 创建特征提取器
        extractor = FeatureExtractor(text_image_service_url=service_url)
        
        # 提取图像特征
        #print(f"Extrating '{image_path}' 's Features...")
        features = extractor.extract_image_features(image_path)
        
        # 打印特征信息
        return features
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def sample_by_diversity(img_list, sample_num):
    """
    从给定的图像列表中，根据多样性采样指定数量的图像。

    Args:
        img_list (list): 包含图像的列表。
        sample_num (int): 需要采样的图像数量。

    Returns:
        list: 包含采样后的图像列表。

    Raises:
        ValueError: 如果sample_num大于img_list的长度，则返回img_list本身。

    """
    n_samples = len(img_list)
    if sample_num >= n_samples:
        return img_list

    features = []
    i = 0
    print(f"### Beginning of Feature Extraction")
    for img in img_list:
        features.append(feature_extract(img))
        i += 1
        if (i % 200 == 0):
            print(f'### {i} images have been extracted... with {len(img_list)} in total')

    print(f"### Complete Feature Extraction, with {len(features)} in total")
    
    features = np.squeeze(np.array(features))

    # 利用K-means聚类
    kmeans = KMeans(n_clusters=sample_num, random_state=42)
    kmeans.fit(features)
    
    centers = kmeans.cluster_centers_    # shape: (sample_num, n_features)
    labels = kmeans.labels_              # 每个样本所属的簇标签
    
    print(f"### Beginning to select {sample_num} samples by diversity")
    selected_indices = []
    # 对于每个簇，选择与聚类中心距离最小的图像索引
    for cluster in range(sample_num):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_features = features[cluster_indices]
        # 计算欧氏距离（也可以用其他距离度量）
        distances = np.linalg.norm(cluster_features - centers[cluster], axis=1)
        rep_index = cluster_indices[np.argmin(distances)]
        selected_indices.append(int(rep_index))
    
    #print(selected_indices)
    return [img_list[i] for i in selected_indices]

if __name__ == "__main__":
    img_list = ["demo_data/raw_data/DS-MVTec/bottle/image/broken_large/000.png", "demo_data/raw_data/DS-MVTec/bottle/image/broken_large/003.png", 
        "demo_data/raw_data/DS-MVTec/bottle/image/contamination/007.png", "demo_data/raw_data/DS-MVTec/capsule/image/crack/001.png", 
        "demo_data/raw_data/DS-MVTec/capsule/image/scratch/003.png", "demo_data/raw_data/DS-MVTec/grid/image/broken/000.png", 
        "demo_data/raw_data/DS-MVTec/leather/image/cut/001.png"]
    sample_num = 4
    sample_img_list = sample_by_diversity(img_list, sample_num)
    print("Original", img_list)
    print("Sample:", sample_img_list)
