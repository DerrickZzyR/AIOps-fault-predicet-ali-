import numpy as np

# 1. 指定你的文件名
file_path = "dataset_SMD_combined.npz"  # 如果是测试集可能是 dataset_test.npz

# 2. 加载 .npz 文件
# allow_pickle=True 是为了保险，不过纯数值数据通常不需要
data = np.load(file_path, allow_pickle=True)

# 3. 查看文件里包含哪些变量名 (应该输出 ['X', 'y'])
print("文件包含的键:", data.files)

# 4. 提取数据
X_data = data['X']
y_data = data['y']

# 5. 打印信息验证
print(f"X (特征) 形状: {X_data.shape}")  # 预期: (样本数, 100)
print(f"y (标签) 形状: {y_data.shape}")  # 预期: (样本数,)

# 查看前 5 个标签
print("前 5 个标签:", y_data[:5])

# 如果你想看第一个样本的波形数据
print("第一个样本数据示例:", X_data[0])