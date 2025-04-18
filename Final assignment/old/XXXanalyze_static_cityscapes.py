import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 分析 Cityscapes 資料集，統計每個類別的像素數
def analyze_cityscapes_pixel_distribution(cityscapes_dir):
    gt_dir = os.path.join(cityscapes_dir, 'gtFine')

    # 定義訓練用的 19 個有效類別
    valid_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 
                     'pole', 'traffic light', 'traffic sign', 'vegetation', 
                     'terrain', 'sky', 'person', 'rider', 'car', 'truck', 
                     'bus', 'train', 'motorcycle', 'bicycle']

    # 更新映射：將原始標籤ID映射到訓練用的類別名稱
    try:
        with open(os.path.join(cityscapes_dir, 'meta', 'labels.json'), 'r') as f:
            labels = json.load(f)
        # 若 labels.json 存在，嘗試利用 trainId 建立映射（排除被標記為 255 的 ignore 類）
        id_to_name = {}
        for label in labels:
            if 'trainId' in label and label['trainId'] != 255:
                id_to_name[label['id']] = label['name']
        # 僅保留在 valid_classes 內的項目
        id_to_name = {k: v for k, v in id_to_name.items() if v in valid_classes}
    except FileNotFoundError:
        print("Labels file not found. Using default training mapping.")
        # 預設映射：僅包含 19 個訓練類別
        id_to_name = {
            7: 'road',
            8: 'sidewalk',
            11: 'building',
            12: 'wall',
            13: 'fence',
            17: 'pole',
            19: 'traffic light',
            20: 'traffic sign',
            21: 'vegetation',
            22: 'terrain',
            23: 'sky',
            24: 'person',
            25: 'rider',
            26: 'car',
            27: 'truck',
            28: 'bus',
            31: 'train',
            32: 'motorcycle',
            33: 'bicycle'
        }

    # 初始化計數，包含有效類別與忽略類別 (ignore)
    pixel_counts = {class_name: 0 for class_name in valid_classes}
    pixel_counts["ignore"] = 0

    # 搜集所有 ground truth 的 semantic segmentation 檔案
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
            if val in id_to_name:
                class_name = id_to_name[val]
                pixel_counts[class_name] += count
            else:
                pixel_counts["ignore"] += count

    return pixel_counts

# 繪製各類別像素百分比分布圖
def plot_pixel_percentage_distribution(pixel_counts, output_path):
    # 僅計算有效類別（19 類）的百分比
    valid_classes = [cls for cls in pixel_counts if cls != "ignore"]
    total_valid_pixels = sum(pixel_counts[cls] for cls in valid_classes)
    
    # 計算每個類別的百分比
    percentages = {cls: (pixel_counts[cls] / total_valid_pixels) * 100 for cls in valid_classes}

    # 依百分比高低排序
    sorted_classes = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    classes, percents = zip(*sorted_classes)

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(range(len(classes)), percents, width=0.6, color='skyblue')

    # 在每個 bar 上方標示百分比數值
    for bar, percent in zip(bars, percents):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{percent:.2f}%', 
                ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_ylabel('Percentage of pixels (%)')
    ax.set_title('Cityscapes Training Set Pixel Distribution by Class')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")
    plt.close()

# 主函數：執行分析
def main():
    cityscapes_dir = "c:/Course_NNCV/NNCV/Final assignment/data/data/Cityscapes"  # 修改為你的 Cityscapes 資料夾路徑
    output_path = "cityscapes_pixel_percentage_distribution_percentage.png"

    pixel_counts = analyze_cityscapes_pixel_distribution(cityscapes_dir)

    total_valid = sum(pixel_counts[cls] for cls in pixel_counts if cls != "ignore")
    print("Total valid pixels counted:", total_valid)
    print("Pixel counts per class:")
    for cls, count in pixel_counts.items():
        print(f"  {cls}: {count}")

    if total_valid == 0:
        print("⚠️ No valid pixel data found. Check dataset and labels.")
        return

    plot_pixel_percentage_distribution(pixel_counts, output_path)

if __name__ == "__main__":
    main()
