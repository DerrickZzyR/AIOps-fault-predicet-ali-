import os
import argparse
import numpy as np
import pandas as pd
import random

def extract_event_based_dataset(df, window_size=100, negative_ratio=3, buffer_size=50):
    """
    基于故障点构建数据集，并带有缓冲带过滤
    :param df: 单个 KPI 的 DataFrame (已经排好序)
    :param window_size: 窗口大小
    :param negative_ratio: 负样本比例
    :param buffer_size: 缓冲带大小 (故障点前后 buffer_size 范围内不采负样本)
    :return: X, y (如果没有提取到数据，返回 None, None)
    """
    # 确保重置索引
    df = df.reset_index(drop=True)
    values = df['value'].values
    labels = df['label'].values
    
    # === 1. 找到所有的故障点 ===
    anomaly_indices = np.where(labels == 1)[0]
    
    # 如果没有故障点，直接返回（或者你可以选择随机采一些负样本，但为了平衡通常跳过）
    if len(anomaly_indices) == 0:
        return None, None

    X = []
    y = []
    
    # === 2. 定义“禁区”掩码 (Forbidden Mask) ===
    # 我们不仅要避开故障点本身，还要避开故障点前后的区域
    # 默认为 False (可采样)，如果是 True 则表示在禁区内
    forbidden_mask = np.zeros(len(values), dtype=bool)
    
    for idx in anomaly_indices:
        # 设定禁区范围：[idx - buffer, idx + buffer]
        # 注意边界检查
        start_buffer = max(0, idx - buffer_size)
        end_buffer = min(len(values), idx + buffer_size)
        forbidden_mask[start_buffer : end_buffer] = True
    
    # === 3. 提取正样本 (故障样本) ===
    valid_anomaly_indices = []
    for idx in anomaly_indices:
        # 必须保证前面有足够的数据切窗口
        if idx >= window_size:
            # 提取 [idx-100, idx)
            window = values[idx - window_size : idx]
            X.append(window)
            y.append(1)
            valid_anomaly_indices.append(idx)
    
    num_positives = len(valid_anomaly_indices)
    if num_positives == 0:
        return None, None

    # === 4. 提取负样本 (正常样本) ===
    # 目标数量
    num_negatives = int(num_positives * negative_ratio)
    
    # 候选池必须满足两个条件：
    # 1. 标签本身是 0 (Normal)
    # 2. 不在禁区内 (Not Forbidden)
    # 3. 前面有足够长度 (idx >= window_size)
    candidate_indices = np.where((labels == 0) & (~forbidden_mask))[0]
    candidate_indices = candidate_indices[candidate_indices >= window_size]
    
    if len(candidate_indices) > 0:
        # 随机采样
        if len(candidate_indices) > num_negatives:
            selected_indices = np.random.choice(candidate_indices, num_negatives, replace=False)
        else:
            selected_indices = candidate_indices # 如果不够，就全拿
            
        for idx in selected_indices:
            window = values[idx - window_size : idx]
            X.append(window)
            y.append(0)
    
    return np.array(X), np.array(y)

def calc_interval(timestamps):
    """根据所有的采样时间计算采样间隔"""
    diff = np.diff(timestamps)
    intervals, counts = np.unique(diff, return_counts=True)
    tmp = np.hstack((intervals.reshape(-1, 1), counts.reshape(-1, 1)))
    return tmp[np.argmax(tmp[:, 1])][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='train')
    args = parser.parse_args()
    
    if args.data_type == 'train':
        all_df = pd.read_csv("data/phase2_train.csv") 
    else:
        all_df = pd.read_hdf("data/phase2_ground_truth.hdf")

    name_dfs = all_df.groupby("KPI ID")
    
    all_X = []
    all_y = []
    
    for name, df in name_dfs:
        df = df.sort_values(by="timestamp")
        # interval = calc_interval(df["timestamp"].values)
        # window_size = int(3600 / interval)  # 1小时的窗口大小
        
        # 调用函数，并接收返回值
        # buffer_size=50 表示：故障点的前50个和后50个时刻，都不作为负样本
        X, y = extract_event_based_dataset(df, window_size=60, negative_ratio=3, buffer_size=30)
        
        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            
    # === 拼接所有 KPI 的数据 ===
    if len(all_X) > 0:
        final_X = np.concatenate(all_X, axis=0)
        final_y = np.concatenate(all_y, axis=0)
        
        print(f"构建完成！")
        print(f"X shape: {final_X.shape}") # (样本数, 100)
        print(f"y shape: {final_y.shape}") # (样本数,)
        print(f"正样本数: {np.sum(final_y==1)}")
        print(f"负样本数: {np.sum(final_y==0)}")
        
        # === 保存 ===
        # 推荐保存为 .npz 格式，因为 X 是多维数组，存 CSV 会很麻烦
        save_path = f"dataset_{args.data_type}.npz"
        np.savez(save_path, X=final_X, y=final_y)
        print(f"数据已保存为 numpy 格式: {save_path}")
        
    else:
        print("未提取到任何有效样本。")

if __name__ == '__main__':
    main()