import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_event_based_dataset(df, window_size=100, negative_ratio=3, buffer_size=50):
    """
    基于故障点构建数据集 (修复了连续异常泄露问题)
    """
    # === 0. 数据预处理 & 归一化 ===
    labels = df['label'].values
    
    # 提取特征列（排除 timestamp 和 label）
    feature_cols = [col for col in df.columns if col not in ['label', 'timestamp']]
    values_raw = df[feature_cols].values

    # 归一化：这对多维度数据至关重要
    scaler = StandardScaler()
    values = scaler.fit_transform(values_raw)
    
    # === 1. 找到“首发”故障点 (Event Start) ===
    # 我们不遍历所有异常点，只找连续异常段的“开头”
    # diff 用于找 0->1 的跳变点
    # pad=0 确保如果第0个就是异常也能被识别
    diff = np.diff(labels, prepend=0) 
    
    # diff == 1 的位置就是异常开始的位置 (Normal -> Anomaly)
    event_start_indices = np.where(diff == 1)[0]
    
    # 如果你也想包含异常开始后的前 K 个点（比如前3个点也能预测），可以微调这里
    # 但严格的“无泄露预测”建议只取 event_start_indices
    target_indices = event_start_indices 

    if len(target_indices) == 0:
        return None, None

    X = []
    y = []
    
    # === 2. 定义禁区 (策略不变) ===
    # 依然使用所有的 anomaly_indices 来画禁区，保证负样本纯净
    anomaly_indices = np.where(labels == 1)[0]
    forbidden_mask = np.zeros(len(labels), dtype=bool)
    
    for idx in anomaly_indices:
        start_buffer = max(0, idx - buffer_size)
        end_buffer = min(len(labels), idx + buffer_size)
        forbidden_mask[start_buffer : end_buffer] = True
    
    # === 3. 提取正样本 (仅针对故障开始点) ===
    # 【修复1】只遍历 target_indices (事件开头)
    valid_pos_count = 0
    for idx in target_indices:
        if idx >= window_size:
            # 窗口：[idx-100, idx)
            # 因为 idx 是故障开始的第一个点，所以 idx-1 是正常的
            # 这样保证了 input window 全是正常数据 -> 预测 idx 处的故障
            window = values[idx - window_size : idx]
            X.append(window)
            y.append(1)
            valid_pos_count += 1
    
    if valid_pos_count == 0:
        return None, None

    # === 4. 提取负样本 ===
    num_negatives = int(valid_pos_count * negative_ratio)
    
    candidate_indices = np.where((labels == 0) & (~forbidden_mask))[0]
    candidate_indices = candidate_indices[candidate_indices >= window_size]
    
    if len(candidate_indices) > 0:
        if len(candidate_indices) > num_negatives:
            selected_indices = np.random.choice(candidate_indices, num_negatives, replace=False)
        else:
            selected_indices = candidate_indices
            
        for idx in selected_indices:
            window = values[idx - window_size : idx]
            X.append(window)
            y.append(0)
    
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/') 
    args = parser.parse_args()
    
    # 获取文件夹下所有的 csv 文件
    # 假设 SMD 的测试集文件都在这里，且文件名包含 'test'
    # 如果你的文件夹里混着 train 和 test，请用 glob 过滤，例如 "*_test.csv"
    search_path = os.path.join(args.data_dir, "*_test.csv")
    file_list = glob.glob(search_path)
    
    if len(file_list) == 0:
        print(f"在 {args.data_dir} 下未找到任何 CSV 文件！")
        return

    print(f"找到 {len(file_list)} 个数据文件，准备合并处理...")
    
    all_X = []
    all_y = []
    total_pos = 0
    total_neg = 0

    for file_path in file_list:
        print(f"正在处理: {os.path.basename(file_path)} ...", end="")
        
        try:
            df = pd.read_csv(file_path)
            
            # 调用提取函数
            # 注意：extract_event_based_dataset 内部已经做了 StandardScaler
            # 所以提取出来的 X 已经是归一化后的数据，可以直接合并
            X, y = extract_event_based_dataset(df, window_size=100, negative_ratio=3)
            
            if X is not None and len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                
                # 统计一下当前文件的样本数
                pos = np.sum(y == 1)
                neg = np.sum(y == 0)
                total_pos += pos
                total_neg += neg
                print(f" 提取到 {len(y)} 个样本 (正:{pos}/负:{neg})")
            else:
                print(" 无有效故障样本 (跳过)")
                
        except Exception as e:
            print(f" [出错]: {e}")

    # === 合并所有机器的数据 ===
    if len(all_X) > 0:
        final_X = np.concatenate(all_X, axis=0)
        final_y = np.concatenate(all_y, axis=0)
        
        print("\n" + "="*30)
        print("所有机器数据处理完成！")
        print(f"总 X shape: {final_X.shape}") # (总样本数, 100, 38)
        print(f"总 y shape: {final_y.shape}")
        print(f"总 正样本数 (故障): {total_pos}")
        print(f"总 负样本数 (正常): {total_neg}")
        print("="*30)
        
        # 保存为一个大的聚合数据集
        save_path = "dataset_SMD_combined.npz"
        np.savez(save_path, X=final_X, y=final_y)
        print(f"聚合数据集已保存至: {save_path}")
    else:
        print("未提取到任何样本，请检查数据或标签。")

if __name__ == '__main__':
    main()