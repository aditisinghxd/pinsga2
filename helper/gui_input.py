import numpy as np
import sys
import os
import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
import matplotlib.pyplot as plt


from helper.genome_to_full_map import genome_to_full_landuse_map
from helper.plot_landuse_share import plot_landuse_share
from helper.timestamp_utils import timestamp_decorator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors




original_map = np.load("processing/original_map.npy")
patchID_map= np.load("processing/patchID.npy")

landuse_labels = ["Cropland 1", "Cropland 2", "Cropland 3", "Cropland 4", "Cropland 5", "Pasture", "Forest", "Urban"]
landuse_colors = ["#fff68f", "#eee685", "#ffd700", "#eec800", "#ccaa01", "#228b22", "#a2cd5a", "#696969"]


cmap = colors.ListedColormap(landuse_colors)
bounds = [0.5 + i for i in range(9)]  # 0.5 to 8.5
norm = colors.BoundaryNorm(bounds, cmap.N)

@timestamp_decorator
def show_pairwise_decision(f1, f2, eta_F, x1, x2):
    #print(">> show_pairwise_decision called", flush=True)
    def on_decision(val):
        nonlocal decision
        decision = val
        #win.destroy()
        win.after(0, win.quit)      # cleanly exit mainloop()
        win.after(1, win.destroy)   # then destroy the window
        

    decision = None
    win = tk.Tk()
    win.title("Pairwise Comparison")
    

    x1_map = genome_to_full_landuse_map(genome=x1, patchID_map=patchID_map, original_map=original_map)
    x2_map = genome_to_full_landuse_map(genome=x2, patchID_map=patchID_map, original_map=original_map)
    x1_chart = plot_landuse_share(x1_map, "Land Use Share: Option A") 
    x2_chart = plot_landuse_share(x2_map, "Land Use Share: Option B") 

    x1_chart.subplots_adjust(left=0, right=1, top=1, bottom=0)
    x2_chart.subplots_adjust(left=0, right=1, top=1, bottom=0)



    # --- Create frame for composite layout ---
    main_frame = tk.Frame(win)
    main_frame.grid()

    # -- Left: Pareto Plot -- Column 0, Row 0
    fig_pf, ax_pf = plt.subplots(figsize=(5, 4))
    #fig_pf.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.9)
    ax_pf.scatter(-eta_F[:, 0], -eta_F[:, 1], color='black', label="Pareto Front")
    ax_pf.scatter(-f1[0], -f1[1], color='blue', label="Option A")
    ax_pf.scatter(-f2[0], -f2[1], color='red', label="Option B")
    ax_pf.set_title("Pareto Front with Highlighted Choices")
    ax_pf.set_xlabel("Crop Yield")
    ax_pf.set_ylabel("Forest Species Richness")
    ax_pf.legend()
    ax_pf.set_xlim(0, 120)
    ax_pf.set_ylim(8, 15)

    canvas_pf = FigureCanvasTkAgg(fig_pf, master=main_frame)
    canvas_pf.draw()
    canvas_pf.get_tk_widget().grid(row=0, column=0)

    # -- Right: Map A and Map B stacked vertically -- Column 1, row 0
    map_frame = tk.Frame(main_frame)
    map_frame.grid(row=0, column=1)

    fig_maps, (ax_map1, ax_map2) = plt.subplots(2, 1, figsize=(4, 4))
    fig_maps.subplots_adjust(left=0, right=1, top=1, bottom=0)


    ax_map1.imshow(x1_map, cmap=cmap, norm=norm)
    ax_map1.set_title("Option A Map")
    ax_map1.axis('off')


    ax_map2.imshow(x2_map, cmap=cmap, norm=norm)
    ax_map2.set_title("Option B Map")
    ax_map2.axis('off')

    canvas_maps = FigureCanvasTkAgg(fig_maps, master=map_frame)
    canvas_maps.draw()
    canvas_maps.get_tk_widget().pack()



    # -- Legend on the right -- Column 2, row 0
    legend_frame = tk.Frame(main_frame)
    legend_frame.grid(row=0, column=2)
    tk.Label(legend_frame, text="Legend", font=('Arial', 10, 'bold')).pack()

    for color, label in zip(landuse_colors, landuse_labels):
        entry = tk.Frame(legend_frame)
        entry.pack(anchor='w', pady=2)
        color_box = tk.Label(entry, width=2, height=1, bg=color)
        color_box.pack(side='left', padx=(0, 5))
        tk.Label(entry, text=label).pack(side='left')


    # -- Pie Charts on far right -- Column 3, row 0
    #x1_chart.set_size_inches(2, 2)  
    #x2_chart.set_size_inches(2, 2)

    pie1_frame = tk.Frame(main_frame)
    pie1_frame.grid(row=0, column=3, sticky='n')

    canvas_pie1 = FigureCanvasTkAgg(x1_chart, master=pie1_frame)
    canvas_pie1.draw()
    canvas_pie1.get_tk_widget().pack()

    pie2_frame = tk.Frame(main_frame)
    pie2_frame.grid(row=0, column=4, sticky='n')

    canvas_pie2 = FigureCanvasTkAgg(x2_chart, master=pie2_frame)
    canvas_pie2.draw()
    canvas_pie2.get_tk_widget().pack()

    # Fitness display under Pareto front and selection buttons-- column 0, row 1
    info_frame = tk.Frame(main_frame)
    info_frame.grid(row=1, column=0)

    tk.Label(info_frame, text=f"Option A: {-f1}").pack()
    tk.Label(info_frame, text=f"Option B: {-f2}").pack()

    tk.Button(info_frame, text="Prefer A", command=lambda: on_decision("a")).pack(side='left')
    tk.Button(info_frame, text="Prefer B", command=lambda: on_decision("b")).pack(side='left')
    tk.Button(info_frame, text="Can't make a decision", command=lambda: on_decision("c")).pack(side='left')

    
    win.mainloop()
    plt.close('all')
    return decision

