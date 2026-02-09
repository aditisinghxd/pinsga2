import numpy as np
import matplotlib.pyplot as plt


def plot_landuse_share(landuse_map, title):
    unique, counts = np.unique(landuse_map, return_counts=True)
    landuse_share = dict(zip(unique, counts))
    
    labels = [
        "Cropland 1", "Cropland 2", "Cropland 3", "Cropland 4",
        "Cropland 5", "Pasture", "Forest", "Urban"
    ]
    colors_list = ["#fff68fff","#eee685ff","#ffd700ff", "#eec800ff", "#ccaa01ff", "#228b22ff", "#a2cd5aff","#696969ff"]

    # Prepare data
    percentages = [landuse_share.get(i, 0) for i in range(1, 9)]
    pie_labels = [labels[i] for i in range(8) if percentages[i] > 0]
    pie_colors = [colors_list[i] for i in range(8) if percentages[i] > 0]
    pie_sizes = [percent for percent in percentages if percent > 0]

    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax.axis('equal')
    ax.set_title(title)
    ax.set_title(title, fontsize=10) 
    return fig