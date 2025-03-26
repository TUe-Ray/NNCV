import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math

# 只統計這 19 個類別，順序依照下列順序
TRAIN_LABELS = [
    (7, 'road'),
    (8, 'sidewalk'),
    (11, 'building'),
    (12, 'wall'),
    (13, 'fence'),
    (17, 'pole'),
    (19, 'traffic light'),
    (20, 'traffic sign'),
    (21, 'vegetation'),
    (22, 'terrain'),
    (23, 'sky'),
    (24, 'person'),
    (25, 'rider'),
    (26, 'car'),
    (27, 'truck'),
    (28, 'bus'),
    (31, 'train'),
    (32, 'motorcycle'),
    (33, 'bicycle')
]

# 根據 TRAIN_LABELS 建立 id 到 name 的對應字典
TRAIN_ID_TO_NAME = {label_id: label_name for label_id, label_name in TRAIN_LABELS}

# 進行統計時僅保留這 19 個類別
def analyze_cityscapes_pixel_distribution(cityscapes_dir):
    gt_dir = os.path.join(cityscapes_dir, 'gtFine')
    
    # 初始化僅針對 19 個類別的像素統計
    pixel_counts = {label_name: 0 for _, label_name in TRAIN_LABELS}

    # 搜尋所有 ground truth semantic segmentation 檔案
    gt_files = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(gt_dir, split)
        if os.path.exists(split_dir):
            for city in os.listdir(split_dir):
                city_dir = os.path.join(split_dir, city)
                if os.path.isdir(city_dir):
                    gt_files.extend(glob.glob(os.path.join(city_dir, '*_labelIds.png')))
    
    print(f"Found {len(gt_files)} ground truth files")
    if len(gt_files) == 0:
        print("⚠️ Warning: No ground truth files found. Check your dataset path.")
    
    # 逐一處理每個 ground truth 檔案
    for i, gt_file in enumerate(gt_files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(gt_files)}")
        label_img = np.array(Image.open(gt_file))
        values, counts = np.unique(label_img, return_counts=True)
        for val, count in zip(values, counts):
            if val in TRAIN_ID_TO_NAME:
                class_name = TRAIN_ID_TO_NAME[val]
                pixel_counts[class_name] += count

    return pixel_counts

# 畫出每個類別的百分比長條圖，並為每個 bar 指定不同的顏色
def plot_pixel_distribution(pixel_counts, output_path):
    total_pixels = sum(pixel_counts.values())
    
    # 依照 TRAIN_LABELS 順序排列
    class_names = [label_name for _, label_name in TRAIN_LABELS]
    percentages = [pixel_counts[name] / total_pixels * 100 for name in class_names]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(len(class_names))
    
    # 透過 colormap 生成不同的顏色
    colors = cm.viridis(np.linspace(0, 1, len(class_names)))
    
    bars = ax.bar(x_pos, percentages, width=0.5, color=colors)
    
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{class_names[i]}\n({percentages[i]:.2f}%)",
                ha='center', va='bottom', rotation=90, fontsize=8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_ylabel('Percentage of pixels (%)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")
    plt.close()

# 輸出文字統計摘要，並附上三個 list：
# 1. Inverse Frequency list
# 2. Median Frequency Balancing list
# 3. Log Scaling list
def output_text_summary(pixel_counts, output_text_path):
    total_pixels = sum(pixel_counts.values())
    class_names = [label_name for _, label_name in TRAIN_LABELS]
    freqs = [pixel_counts[name] / total_pixels for name in class_names]
    
    # 1. Inverse Frequency list：1 / freq
    inverse_freq = [1.0 / f if f > 0 else 0 for f in freqs]
    
    # 2. Median Frequency Balancing list：median(freqs) / freq
    median_freq = np.median(freqs)
    median_balancing = [median_freq / f if f > 0 else 0 for f in freqs]
    
    # 3. Log Scaling list：1 / log(1.02 + freq)
    log_scaling = [1.0 / math.log(1.02 + f) for f in freqs]
    
    lines = []
    lines.append("Cityscapes pixel distribution summary (19 training classes):\n")
    lines.append(f"Total pixels (for these 19 classes): {total_pixels}\n")
    lines.append("Class-wise statistics:")
    for name, f, count in zip(class_names, freqs, [pixel_counts[n] for n in class_names]):
        lines.append(f"  {name}: {count} pixels, {f*100:.2f}%")
    
    lines.append("\nPython lists for loss weighting (order follows the class order):")
    lines.append(f"Inverse Frequency list: {inverse_freq}")
    lines.append(f"Median Frequency Balancing list: {median_balancing}")
    lines.append(f"Log Scaling list: {log_scaling}")
    
    with open(output_text_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Text summary saved to {output_text_path}")

# Main function to run the analysis and output results
def main():
    cityscapes_dir = "c:/Course_NNCV/NNCV/Final assignment/data/data/Cityscapes"  # 修改成你的 Cityscapes 資料夾路徑
    output_image_path = "cityscapes_pixel_distribution.png"
    output_text_path = "cityscapes_pixel_distribution.txt"
    
    pixel_counts = analyze_cityscapes_pixel_distribution(cityscapes_dir)
    if sum(pixel_counts.values()) == 0:
        print("⚠️ No valid pixel data found. Check dataset and labels.")
        return
    
    plot_pixel_distribution(pixel_counts, output_image_path)
    output_text_summary(pixel_counts, output_text_path)

if __name__ == "__main__":
    main()
