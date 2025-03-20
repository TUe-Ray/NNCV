import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from collections import Counter
import json

# Function to process the Cityscapes dataset
def analyze_cityscapes_pixel_distribution(cityscapes_dir):
    # Define paths for the different splits
    gt_dir = os.path.join(cityscapes_dir, 'gtFine')
    
    # Cityscapes class definitions
    class_info = {
        # Main categories
        'flat': ['road', 'sidewalk', 'parking', 'rail track'],
        'construction': ['building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel'],
        'nature': ['vegetation', 'terrain'],
        'vehicle': ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'caravan', 'trailer'],
        'sky': ['sky'],
        'object': ['pole', 'traffic light', 'traffic sign', 'pole group'],
        'human': ['person', 'rider'],
        'void': ['static', 'dynamic', 'ground']
    }
    
    # Get class names from cityscape labels
    try:
        with open(os.path.join(cityscapes_dir, 'meta', 'labels.json'), 'r') as f:
            labels = json.load(f)
        
        id_to_name = {}
        for label in labels:
            id_to_name[label['id']] = label['name']
    except FileNotFoundError:
        print("Labels file not found. Using default class IDs.")
        # Define default mappings if labels.json is not found
        id_to_name = {
            0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 
            5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
            9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
            14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
            19: 'static', 20: 'dynamic', 21: 'ground', 22: 'parking', 23: 'rail track',
            24: 'guard rail', 25: 'bridge', 26: 'tunnel', 27: 'caravan', 28: 'trailer',
            29: 'pole group'
        }
    
    # Initialize counters
    pixel_counts = {}
    for category in class_info:
        for class_name in class_info[category]:
            pixel_counts[class_name] = 0
    
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
    
    # Process each ground truth file
    for i, gt_file in enumerate(gt_files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(gt_files)}")
            
        # Load the label image
        label_img = np.array(Image.open(gt_file))
        
        # Count pixel occurrences
        values, counts = np.unique(label_img, return_counts=True)
        
        # Map pixel values to class names and update counts
        for val, count in zip(values, counts):
            if val in id_to_name:
                class_name = id_to_name[val]
                if class_name in pixel_counts:
                    pixel_counts[class_name] += count
    
    # Organize data by categories
    category_data = {}
    for category, classes in class_info.items():
        category_data[category] = {c: pixel_counts.get(c, 0) for c in classes if c in pixel_counts}
    
    return category_data

# Function to plot the distribution
def plot_pixel_distribution(category_data, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Colors for each category
    category_colors = {
        'flat': '#8A2BE2',
        'construction': '#808080',
        'nature': '#90EE90',
        'vehicle': '#4169E1',
        'sky': '#87CEEB',
        'object': '#FFFF00',
        'human': '#FF69B4',
        'void': '#2F4F4F'
    }
    
    # Position for each category
    category_positions = list(range(len(category_data)))
    
    # Plot each category with its classes
    x_ticks = []
    x_positions = []
    x_labels = []
    
    for i, (category, classes) in enumerate(category_data.items()):
        # Determine x position
        x_pos = category_positions[i]
        x_ticks.append(x_pos + 0.5)
        
        # Sort classes by count in descending order
        sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
        
        # Plot each class in the category
        for j, (class_name, count) in enumerate(sorted_classes):
            bar_pos = x_pos + j*0.15
            x_positions.append(bar_pos)
            x_labels.append(class_name)
            
            # Add superscript for annotations
            superscript = ""
            if "instance-level" in class_name:
                superscript = "¹"
            elif "ignored" in class_name:
                superscript = "²"
            
            ax.bar(bar_pos, count, width=0.1, color=category_colors[category], 
                   alpha=1.0 - j*0.05 if j < 10 else 0.5)
            
            # Add text label above the bar
            ax.text(bar_pos, count*1.05, class_name + superscript, 
                    ha='center', va='bottom', rotation=90, fontsize=8)
    
    # Set up the plot
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(category_data.keys())
    ax.set_ylabel('number of pixels')
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(1e5, 1e10)
    
    # Add legend for superscripts
    ax.text(0.98, 0.98, "¹ instance-level annotations are available", 
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.text(0.98, 0.95, "² ignored for evaluation", 
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

# Main function to run the analysis
def main():
    # Path to the Cityscapes dataset on Snellius
    # Change this to match your Cityscapes dataset location
    cityscapes_dir = "/path/to/cityscapes"
    
    # Output path for the plot
    output_path = "cityscapes_pixel_distribution.png"
    
    # Run the analysis
    category_data = analyze_cityscapes_pixel_distribution(cityscapes_dir)
    
    # Plot the results
    plot_pixel_distribution(category_data, output_path)

if __name__ == "__main__":
    main()