import numpy as np
from helper.genome_to_full_map import genome_to_full_landuse_map

patchID_map= np.load("processing/patchID.npy")
original_map = np.load("processing/original_map.npy")

def crop_yield(land_use_genome, soil_fertility_map):

    back_to_landusemap= genome_to_full_landuse_map(genome=land_use_genome, patchID_map=patchID_map, original_map=original_map)
    back_to_landusemap= back_to_landusemap.flatten()

    production_intensity_map = np.zeros_like(back_to_landusemap, dtype=int)
    for i in range(1, 6):  # Cropland 1-5
        production_intensity_map[back_to_landusemap == i] = i  
    
    crop_yield_map = np.log(production_intensity_map * (1 + soil_fertility_map))
    crop_yield_map[np.isnan(crop_yield_map) | np.isneginf(crop_yield_map)] = 0

    total_crop_yield = np.sum(crop_yield_map)
    return total_crop_yield



def forest_species_richness(land_use_genome):

    back_to_landusemap= genome_to_full_landuse_map(genome=land_use_genome, patchID_map=patchID_map, original_map=original_map)
    back_to_landusemap= back_to_landusemap.flatten()

    forest_area = np.sum(back_to_landusemap == 7)
    
    c = 5 
    z = 0.2
    forest_species_richness = c * (forest_area ** z)
    return forest_species_richness