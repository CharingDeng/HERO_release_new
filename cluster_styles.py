import torch
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
import json
from utils.motion_process import recover_from_ric

def extract_features(motion_data, joints_num=22):
    # 确保输入是 PyTorch 张量
    if isinstance(motion_data, np.ndarray):
        motion_data_tensor = torch.from_numpy(motion_data).float()
    else:
        motion_data_tensor = motion_data
    
    seq_len = motion_data_tensor.shape[0]
    
    # 动态特征：速度、加速度统计量（使用 NumPy）
    motion_data_np = motion_data_tensor.numpy()
    if seq_len < 3:
        dyn_feats = np.zeros(263 * 4)
    else:
        velocity = motion_data_np[1:] - motion_data_np[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        dyn_feats = np.concatenate([
            velocity.mean(axis=0), velocity.std(axis=0),
            acceleration.mean(axis=0), acceleration.std(axis=0)
        ])
    
    # 时空特征：关节运动范围（使用 PyTorch 张量）
    try:
        joints = recover_from_ric(motion_data_tensor, joints_num)
        joints_np = joints.numpy()
        
        arm_joints = joints_np[:, [5, 6, 7, 9, 10, 11]]
        leg_joints = joints_np[:, [12, 13, 14, 15, 16, 17]]
        arm_range = arm_joints.max(axis=0) - arm_joints.min(axis=0)
        leg_range = leg_joints.max(axis=0) - leg_joints.min(axis=0)
        spatial_feats = np.concatenate([arm_range.flatten(), leg_range.flatten()])
        
        return np.concatenate([dyn_feats, spatial_feats])
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None

# 其余代码保持不变...
def cluster_styles(motion_dir, style_dir, num_clusters=4):
    all_features = []
    motion_paths = []

    # 第二步：遍历所有动作文件，提取特征
    for action_type in os.listdir(motion_dir):
        action_motion_dir = os.path.join(motion_dir, action_type)
        action_style_dir = os.path.join(style_dir, action_type)
        
        for motion_file in os.listdir(action_motion_dir):
            motion_path = os.path.join(action_motion_dir, motion_file)
            motion_data = np.load(motion_path)
            feats = extract_features(motion_data)
            if feats is not None:
                all_features.append(feats)
                motion_paths.append((action_style_dir, motion_file))  # 记录保存路径和文件名

    # 异常检测：使用IsolationForest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso_forest.fit_predict(all_features)
    all_features = [all_features[i] for i in range(len(outliers)) if outliers[i] == 1]
    motion_paths = [motion_paths[i] for i in range(len(outliers)) if outliers[i] == 1]

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    # PCA降维：保留主要特征
    pca = PCA(n_components=6)  # 保留主要方差
    features_pca = pca.fit_transform(features_scaled)

    # KMeans聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_pca)

    # 聚类效果评估
    sil_score = silhouette_score(features_pca, labels)
    db_score = davies_bouldin_score(features_pca, labels)
    print(f"Silhouette Score: {sil_score}, Davies-Bouldin Score: {db_score}")

    # 第三步：为每个动作生成独立的风格标签JSON文件
    for (action_style_dir, motion_file), label in zip(motion_paths, labels):
        json_path = os.path.join(action_style_dir, motion_file.replace('.npy', '.json'))
        with open(json_path, 'w') as f:
            json.dump({"style_label": int(label)}, f)  # 保存风格标签（0-3）
        print(f"已生成风格标签：{json_path}")

    print(f"所有动作的风格标签已生成，共 {len(motion_paths)} 个文件")

if __name__ == "__main__":
    data_root = "../Data/VIMO"  # 替换为你的实际路径
    motion_dir = os.path.join(data_root, "vector_263")
    style_dir = os.path.join(data_root, "style_labels")  # 与vector_263并列的风格标签文件夹
    num_clusters = 4  # 4种风格：活泼/轻快/含蓄/沉稳
    
    # 确保style_labels目录结构存在
    if not os.path.exists(style_dir):
        os.makedirs(style_dir)
    
    # 执行聚类并生成风格标签
    cluster_styles(motion_dir, style_dir, num_clusters)
