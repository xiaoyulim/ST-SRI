import scipy.io
import os

data_dir = "./data"
mat_file = os.path.join(data_dir, "S1_E2_A1.mat")

print(f"Loading: {mat_file}")
mat = scipy.io.loadmat(mat_file)

print("\n=== 字段列表 (非私有) ===")
fields = [k for k in mat.keys() if not k.startswith('__')]
print(fields)

print("\n=== 各字段详情 ===")
for k in fields:
    arr = mat[k]
    if hasattr(arr, 'shape'):
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    else:
        print(f"  {k}: {type(arr)}")

print("\n=== 检查 force/kinematic 相关字段 ===")
force_keywords = ['force', 'kinematic', 'emg_gt', 'glove', 'resistive', 'joint']
for k in fields:
    lower_k = k.lower()
    if any(w in lower_k for w in force_keywords):
        arr = mat[k]
        print(f"\n找到可疑字段: {k}")
        print(f"  shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        if arr.size < 1000:
            print(f"  内容预览: {arr.flatten()[:20]}")
        else:
            print(f"  内容范围: min={arr.min():.4f}, max={arr.max():.4f}")