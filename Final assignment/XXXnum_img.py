import os

def count_cityscape_images(base_path='C:\\Course_NNCV\\NNCV\\Final assignment\\data\\data\\cityscapes\\gtFine'):
    counts = {}
    for split in ['train', 'val']:
        split_path = os.path.join(base_path, split)
        print(f"\n🔍 Scanning: {split_path}")  # 新增印出
        img_count = 0
        for root, _, files in os.walk(split_path):
            print(f"📁 Folder: {root} -> {len(files)} files")
            for f in files:
                if f.endswith('.json'):
                    img_count += 1
        counts[split] = img_count
    return counts

if __name__ == "__main__":
    result = count_cityscape_images()
    print("\n📊 Cityscapes Image Counts:")
    print(f"Training images: {result.get('train', 0)}")
    print(f"Validation images: {result.get('val', 0)}")
