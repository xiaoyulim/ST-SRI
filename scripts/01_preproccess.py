# 文件名: 01_preprocess.py
import scipy.io
import numpy as np
import os
import glob

# ================= 配置区域 =================
# 1. 你的原始 .mat 文件在哪里？
RAW_DATA_DIR = "./data"
# 2. 转换后的 .npy 存哪里？
OUTPUT_DIR = "./data"
# 3. 目标练习 (NinaPro DB2 E2 包含核心手势)
TARGET_EXERCISE = 'E1'


# ===========================================

def convert_mat_to_npy():
    print(f">>> [Step 1] 开始转换数据: .mat -> .npy")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 遍历 S1 到 S40
    count = 0
    for sub in range(1, 41):
        # 搜索文件名模式，例如 S1_E2_A1.mat
        file_pattern = os.path.join(RAW_DATA_DIR, f"S{sub}_{TARGET_EXERCISE}_*.mat")
        files = glob.glob(file_pattern)

        if not files:
            print(f"  [跳过] S{sub}: 在 {RAW_DATA_DIR} 下没找到 E2 数据文件")
            continue

        mat_path = files[0]
        try:
            print(f"  正在处理: {os.path.basename(mat_path)} ...")
            mat = scipy.io.loadmat(mat_path)

            # 提取信号 (emg) 和 标签 (stimulus/restimulus)
            # NinaPro DB2 通常用 'emg' 和 'restimulus'
            if 'emg' in mat:
                emg = mat['emg']  # [Time, 12]
            else:
                print(f"    Error: 没找到 'emg' 字段")
                continue

            if 'restimulus' in mat:
                label = mat['restimulus']
            elif 'stimulus' in mat:
                label = mat['stimulus']
            else:
                print(f"    Error: 没找到标签字段")
                continue

            # 数据类型转换 (压缩体积)
            emg = emg.astype(np.float32)
            label = label.astype(np.int64).squeeze()

            # 保存
            np.save(os.path.join(OUTPUT_DIR, f"S{sub}_data.npy"), emg)
            np.save(os.path.join(OUTPUT_DIR, f"S{sub}_label.npy"), label)

            count += 1

        except Exception as e:
            print(f"  [异常] S{sub}: {e}")

    print(f"\n>>> 转换完成！共处理了 {count} 个受试者。")
    print(">>> 现在你可以运行训练脚本了。")


if __name__ == "__main__":
    convert_mat_to_npy()