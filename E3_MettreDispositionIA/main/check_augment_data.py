import os 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter


def analyze_class_distribution(data_dir: str):
    """
    Compte le nombre d'images par classe dans le dossier de données.

    Args:
        data_dir: Dossier racine contenant les sous-dossiers de classes 

    Returns:
        Un Counter avec {classe: nombre d'images}
    """
    class_counts = Counter()

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue 

        num_images = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        class_counts[class_name] = num_images

    return class_counts
            

def export_class_distribution(counts: Counter, output_path="class_distribution.csv"):
    """
    Exporte la distribution des classes dans un fichier CSV.

    Args:
        counts: Counter avec {classe: nombre d'images}
        output_path: Chemin du fichier CSV de sortie (par défaut: "class_distribution.csv")
    """
    df = pd.DataFrame(counts.items(), columns=["Classe", "Nombre"])
    df = df.sort_values(by="Nombre", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Distribution des classes exportée dans {output_path}")



def plot_class_distribution(counts: Counter, title="Répartition des classes", save_path="class_distribution.png"):
    """
    Affiche et enregistre un graphique de la distribution des classes.

    Args:
        counts: Counter avec {classe: nombre d'images}
        title: Titre du graphique
        save_path: Chemin du fichier PNG de sortie (par défaut: "class_distribution.png")
    """
    sorted_counts = counts.most_common()
    classes = [cls for cls, _ in sorted_counts]
    freqs = [count for _, count in sorted_counts]

    plt.figure(figsize=(12,6))
    bars = plt.bar(classes, freqs, color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'images")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Distribution des classes enregistrée dans {save_path}")
    else:
        plt.show()
        
    
if __name__ == "__main__":
    from collections import Counter

    input_dir = "data/augmented_gravures"
    counts = analyze_class_distribution(input_dir)

    export_class_distribution(counts, "analyse_files/class_distribution.csv")
    plot_class_distribution(counts, save_path="analyse_files/class_distribution.png")       
    
    
