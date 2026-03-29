import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 修复环境
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
CSV_PATH = "./results/e5_faithfulness/e5_detailed_results.csv"
SAVE_PATH = "results/e5_faithfulness/e5_1.png"


# ========================================

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def generate_table_image():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到文件: {CSV_PATH}")
        return

    # 1. 读取数据
    df = pd.read_csv(CSV_PATH)

    # 2. 计算统计摘要 (Mean ± Std)
    numeric_cols = ['peak_ms', 'acc_base', 'drop_recent', 'drop_emd', 'ratio']

    mean_vals = df[numeric_cols].mean()
    std_vals = df[numeric_cols].std()

    # 3. 格式化数据 (保留小数位)
    df_styled = df.copy()
    df_styled['subject'] = df_styled['subject'].apply(lambda x: f"S{int(x)}")

    # 对每一列进行特定的格式化
    df_styled['peak_ms'] = df['peak_ms'].apply(lambda x: f"{x:.1f}")
    df_styled['acc_base'] = df['acc_base'].apply(lambda x: f"{x:.1f}")
    df_styled['drop_recent'] = df['drop_recent'].apply(lambda x: f"{x:.2f}")
    df_styled['drop_emd'] = df['drop_emd'].apply(lambda x: f"{x:.2f}")
    df_styled['ratio'] = df['ratio'].apply(lambda x: f"{x:.2f}")

    # 4. 构建汇总行
    summary_row = {
        'subject': 'Mean ± SD',
        'peak_ms': f"{mean_vals['peak_ms']:.1f} ± {std_vals['peak_ms']:.1f}",
        'acc_base': f"{mean_vals['acc_base']:.1f} ± {std_vals['acc_base']:.1f}",
        'drop_recent': f"{mean_vals['drop_recent']:.2f} ± {std_vals['drop_recent']:.2f}",
        'drop_emd': f"{mean_vals['drop_emd']:.2f} ± {std_vals['drop_emd']:.2f}",
        'ratio': f"{mean_vals['ratio']:.2f} ± {std_vals['ratio']:.2f}"
    }

    # 将汇总行添加到 DataFrame 底部
    df_final = pd.concat([df_styled, pd.DataFrame([summary_row])], ignore_index=True)

    # 5. 重命名列头 (让它更好看)
    column_mapping = {
        'subject': 'Subject',
        'peak_ms': 'Peak (ms)',
        'acc_base': 'Base Acc (%)',
        'drop_recent': 'Drop Recent (%)',
        'drop_emd': 'Drop EMD (%)',
        'ratio': 'Faithfulness Ratio'
    }
    df_final = df_final.rename(columns=column_mapping)

    # 6. 绘图并保存
    # 动态调整图片高度，防止太长或太短
    rows = len(df_final)
    plt.figure(figsize=(10, rows * 0.4 + 1.5), dpi=200)  # 0.4 是单行高度系数
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    render_mpl_table(df_final, header_columns=0, col_width=2.5, ax=ax)

    plt.title("E5: Quantitative Faithfulness Results (Summary)", fontsize=14, weight='bold', pad=20)
    plt.savefig(SAVE_PATH, bbox_inches='tight', pad_inches=0.2)
    print(f"✅ 表格图片已生成: {SAVE_PATH}")


if __name__ == "__main__":
    generate_table_image()