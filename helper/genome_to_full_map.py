import numpy as np

def genome_to_full_landuse_map(genome, patchID_map, original_map):
    land_use_map = np.full_like(patchID_map, -2)

    for i in range(patchID_map.shape[0]):
        for j in range(patchID_map.shape[1]):
            patch_id = patchID_map[i, j]
            if patch_id == 0:
                # Static cell → copy from original map
                land_use_map[i, j] = original_map[i, j]
            else:
                # Dynamic patch → assign land use from genome
                land_use_map[i, j] = genome[patch_id - 1]
    
    return land_use_map