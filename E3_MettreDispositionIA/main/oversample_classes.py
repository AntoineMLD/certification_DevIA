import os
import random
import shutil
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

def analyze_class_distribution(data_dir):
    counts = Counter()
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        num_images = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        counts[class_name] = num_images
    return counts

def plot_class_distribution(counts, title="Répartition finale des classes", save_path=None):
    sorted_counts = counts.most_common()
    classes = [cls for cls, _ in sorted_counts]
    freqs = [count for _, count in sorted_counts]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, freqs, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Graphique sauvegardé sous : {save_path}")
    else:
        plt.show()

def oversample_directory(input_dir: str, output_dir: str, min_threshold: int = 80):
    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_path):
            continue

        os.makedirs(class_output_path, exist_ok=True)

        images = [
            img for img in os.listdir(class_input_path)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        num_images = len(images)
        print(f"Classe '{class_name}': {num_images} images")

        for img_name in images:
            src = os.path.join(class_input_path, img_name)
            dst = os.path.join(class_output_path, img_name)
            shutil.copyfile(src, dst)

        if num_images < min_threshold:
            deficit = min_threshold - num_images
            print(f"  Oversampling: besoin de {deficit} duplications")

            for i in range(deficit):
                img_to_duplicate = random.choice(images)
                src_path = os.path.join(class_input_path, img_to_duplicate)
                img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)

                if img is None:
                    print(f"  ⚠️ Erreur lecture {img_to_duplicate}")
                    continue

                new_name = f"{Path(img_to_duplicate).stem}_dup_{i}.png"
                dst_path = os.path.join(class_output_path, new_name)
                cv2.imencode('.png', img)[1].tofile(dst_path)

            print(f"  Classe '{class_name}' portée à {min_threshold} images")
        else:
            print(" Aucune duplication nécessaire")

    print("\n🎉 Oversampling terminé.")

if __name__ == "__main__":
    input_dir = "data/augmented_gravures"
    output_dir = "data/oversampled_gravures"

    oversample_directory(input_dir, output_dir, min_threshold=80)
    counts = analyze_class_distribution(output_dir)
    plot_class_distribution(counts, save_path="analyse_files/oversampled_distribution.png")
