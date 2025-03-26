import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Function to process the Cityscapes dataset
def analyze_cityscapes_pixel_distribution(cityscapes_dir):
    gt_dir = os.path.join(cityscapes_dir, 'gtFine')

    # Cityscapes class definitions
    class_info = {
        'flat': ['road', 'sidewalk', 'parking', 'rail track'],
        'construction': ['building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel'],
        'nature': ['vegetation', 'terrain'],
        'vehicle': ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'caravan', 'trailer'],
        'sky': ['sky'],
        'object': ['pole', 'traffic light', 'traffic sign', 'pole group'],
        'human': ['person', 'rider'],
        'void': ['static', 'dynamic', 'ground']
    }

    # Load label mapping
    try:
        with open(os.path.join(cityscapes_dir, 'meta', 'labels.json'), 'r') as f:
            labels = json.load(f)

        id_to_name = {label['id']: label['name'] for label in labels}
    except FileNotFoundError:
        print("Labels file not found. Using default class IDs.")
        # 正確的 labelId 對 className 映射（含 void、unlabeled 類別）
        id_to_name = {
            0: 'unlabeled', 1: 'ego vehicle', 2: 'rectification border', 3: 'out of roi',
            4: 'static', 5: 'dynamic', 6: 'ground', 7: 'road', 8: 'sidewalk', 9: 'parking',
            10: 'rail track', 11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail',
            15: 'bridge', 16: 'tunnel', 17: 'pole', 18: 'pole group', 19: 'traffic light',
            20: 'traffic sign', 21: 'vegetation', 22: 'terrain', 23: 'sky', 24: 'person',
            25: 'rider', 26: 'car', 27: 'truck', 28: 'bus', 29: 'caravan', 30: 'trailer',
            31: 'train', 32: 'motorcycle', 33: 'bicycle'
        }


    # Initialize counters
    pixel_counts = {class_name: 0 for category in class_info for class_name in class_info[category]}

    # Find all ground truth semantic segmentation files
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

    # Process each ground truth file
    for i, gt_file in enumerate(gt_files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(gt_files)}")
            
        label_img = np.array(Image.open(gt_file))
        values, counts = np.unique(label_img, return_counts=True)

        for val, count in zip(values, counts):
            if val in id_to_name:
                class_name = id_to_name[val]
                if class_name in pixel_counts:
                    pixel_counts[class_name] += count

    # Organize data by categories
    category_data = {category: {c: pixel_counts[c] for c in classes if pixel_counts[c] > 0} 
                     for category, classes in class_info.items()}

    return category_data

# Function to plot the distribution
def plot_pixel_distribution(category_data, output_path, top_n=10):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(16, 6))

    category_colormaps = {
        'flat': cm.Purples,
        'construction': cm.Greys,
        'nature': cm.Greens,
        'vehicle': cm.Blues,
        'sky': cm.Blues,
        'object': cm.YlOrBr,
        'human': cm.Reds,
        'void': cm.cividis
    }

    x_pos = 0
    x_ticks, x_labels = [], []

    for category, classes in category_data.items():
        num_classes = len(classes)
        colormap = category_colormaps[category]
        vmin, vmax = 0.3, 1.0  # 避免太亮的前段色
        norm = mcolors.Normalize(vmin=0, vmax=max(num_classes - 1, 1))
        colors = [colormap(vmin + (vmax - vmin) * i / max(num_classes - 1, 1)) for i in range(num_classes)]


        sorted_items = sorted(classes.items(), key=lambda x: x[1], reverse=True)
        
        for i, (class_name, count) in enumerate(sorted_items[:top_n]):
            bar_pos = x_pos
            ax.bar(bar_pos, count, width=0.4, color=colors[i])
            ax.text(bar_pos, count * 1.05, class_name, ha='center', va='bottom', rotation=90, fontsize=8)
            x_labels.append(class_name)
            x_ticks.append(bar_pos)
            x_pos += 0.6  # avoid overlapping bars
        
        x_pos += 1.0  # space between categories

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_ylabel('Number of pixels')
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=1e5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")
    plt.close()


# Main function to run the analysis
def main():
    cityscapes_dir = "c:/Course_NNCV/NNCV/Final assignment/data/data/Cityscapes"  # 修改成你的 Cityscapes 資料夾
    output_path = "cityscapes_pixel_distribution.png"

    category_data = analyze_cityscapes_pixel_distribution(cityscapes_dir)

    if sum(sum(category.values()) for category in category_data.values()) == 0:
        print("⚠️ No valid pixel data found. Check dataset and labels.")
        return

    plot_pixel_distribution(category_data, output_path)

if __name__ == "__main__":
    main()
