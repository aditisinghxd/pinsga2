import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

from collections import Counter
from pymoo.algorithms.moo.pinsga2 import PINSGA2, AutomatedDM
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from datetime import datetime
from helper.timestamp_utils import print_timestamp
from helper.genome_to_full_map import genome_to_full_landuse_map
from helper.toy_models import crop_yield, forest_species_richness

map_size = "10x10"          # SELECT MAP SIZE: 10x10/20x20/30x30/40x40/100x100
pop_size = 40               # Population Size
max_gen = 200               # Maximum number of generations
dm_period = 50              # DM Period: after how many generations
points= 5                   # Number of points compared during a DM call
type_of_dm = 'manual'    # Type of DM: 'manual' or 'automated' 
x, y = 90,10                # Reference point for the autmated dm : x - Crop Yield; # y - Forest Species Richness
seeds = 6111321             #6111321



max_range = 8            # max range of land-use types
four_neighbours = 'True' # Type of Clustering for patch-ID creation

wrkDir = os.path.abspath(".")
trans_file = 'input/transition_matrix.txt'
minmax_file = "input/min_max.txt"
map_file = f"input/{map_size}/land_use.asc"
soil_fer = f"input/{map_size}/soil_fertility.asc"

static_elements = []
nonstatic_elements = []
trans_matrix = []

trans_matrix = np.genfromtxt(os.path.join(wrkDir, trans_file), dtype=int, filling_values='-1')


#Maximum value of the objective function depending on the map size:
max_values_dict = {
    "10x10": np.array([203.0, 12.56]),
    "20x20": np.array([812.0, 16.57]),
    "30x30": np.array([1823.0, 19.49]),
    "40x40": np.array([3278.0, 21.87]),
    "100x100": np.array([20300.0, 31.55]),
}

max_values = max_values_dict.get(map_size)
max_yield, max_species = max_values

if max_values is None:
    raise ValueError(f"Unsupported map size: {map_size}")


def determine_static_classes(trans_matrix, max_range):
	"""This function determines all classes which are excluded from optimization (static elements) 
		and returns arrays with the indices of static and non static elements.
		
		input:
			trans_matrix holding the land use transition constraints
			max_range is the maximum number of possible land use options
	""" 
	
	# identify all classes where column and row elements are zero (apart from diagonal elements)
	# that means that all patches of these classes cannot be converted
	static_elements = []
	nonstatic_elements = []
	# filter columns which fulfill the condition
	ones = 0
	# row-wise check for ones
	for row in range(1,trans_matrix.shape[0]):
		for col in range(1,trans_matrix.shape[1]):
			if trans_matrix[row][col] == 1:
				ones += 1
		# mark the candidate as static or non static element (row is checked)
		# if ones for row = 1 or max_range < land use index of trans_matrix
		if ones==1 or trans_matrix[row][0] > max_range:
			static_elements.append(trans_matrix[row][0])
		else:
			nonstatic_elements.append(trans_matrix[row][0])
		ones = 0
			
	# column-wise check for ones
	ones = 0
	index = 0
	if len(static_elements) != 0:
		for col in range(1,trans_matrix.shape[1]):
			if index < len(static_elements) and static_elements[index] <= max_range:
				if trans_matrix[0][col] == static_elements[index]:
					for row in range(1,trans_matrix.shape[0]):
						if trans_matrix[row][col] == 1:
							ones += 1
					if ones!=1:
						# index remains as it is for the next round 
						# because of length reduction from static_element
						nonstatic_elements.append(static_elements[index])
						del static_elements[index]
					else:
						index += 1
					ones = 0
				if len(static_elements) == 0:
					break
	
	return static_elements, nonstatic_elements
	
static_elements, nonstatic_elements = determine_static_classes(trans_matrix, max_range)

# Reading maps: Land-Use map and Soil Fertility map
def read_ascii_map(file):
    """Read an ascii file.
    Return the map as matrix and header data as array.

    input data: ascii file
    """

    # read header information in the header array
    header_all = open(file, 'rb').readlines()[0:6]
    header = []
    for line in header_all:
        line_split = line.split()
        header.append(line_split[1])

    if 'soil_fer' in file:
        print('Soil Fertiliy Map')
        dtype = float
    else:
        print('Land-Use Map')
        dtype = int

    # read the map in the map matrix
    map = np.genfromtxt(file, dtype=dtype, skip_header=6, filling_values='-1')

    print("map \n%s" % map)
    return header, header_all, map


if map_file != 'None': 
		header, header_all, map = read_ascii_map(os.path.join(wrkDir, map_file))
np.save('processing\\original_map',map)

if soil_fer != 'None': 
		_, _, soil_map = read_ascii_map(os.path.join(wrkDir, soil_fer))

soil_fertility_map = soil_map
soil_fertility_map = soil_fertility_map.flatten()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Patch ID map creation 
def getNbh(col, row, ncols, nrows, four_neighbours):
    """Determine the neighboring cells of the cell (col,row) and
       return the coordinates as arrays separated in nbhs_col and nbhs_row.
       The combination of the elements gives the coordinates of the neighbouring cells.
       
       input:
           col and row are coordinates of the reviewed element
           ncols, nrows are numbers of rows and columns in the map
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """
    
    # assuming that a cell in the center has 8 neighbouring cells
    if four_neighbours == 'False':
        # cell is no edge cell
        if col > 0 and row > 0 and row < nrows -1 and col < ncols -1:
            nbhs_col = [x + col for x in[-1, -1, -1,  0, 0,  1, 1, 1]]
            nbhs_row = [x + row for x in[-1,  0,  1, -1, 1, -1, 0, 1]]
        # cell is a left edge element but no corner element
        elif col == 0 and row > 0 and row < nrows -1:
            nbhs_col= [x + col for x in[0, 1, 1, 0, 1]]
            nbhs_row= [x + row for x in[-1, -1, 0, 1, 1]]   
        # cell is a right edge element but no corner element
        elif col == ncols -1 and row > 0 and row < nrows -1:
            nbhs_col= [x + col for x in[-1, -1, -1,  0, 0]]
            nbhs_row= [x + row for x in[-1,  0,  1, -1, 1]]
        # cell is an upper edge element but no corner element
        elif row == 0 and col > 0 and col < ncols -1:
            nbhs_col= [x + col for x in[-1, -1,  0, 1, 1 ]]
            nbhs_row= [x + row for x in[ 0,  1, 1, 0, 1 ]]
        # cell is a bottom edge element but no corner element    
        elif row == nrows -1 and col > 0 and col < ncols -1:
            nbhs_col= [x + col for x in[-1, -1,  0,  1, 1 ]]
            nbhs_row= [x + row for x in[ -1, 0, -1, -1, 0 ]] 
        # cell is in the left upper corner
        elif col == 0 and row == 0:
            nbhs_col= [x + col for x in[ 0, 1, 1]]
            nbhs_row= [x + row for x in[ 1, 0, 1]]
        # cell is in the left bottom corner
        elif col == 0 and row == nrows -1:
            nbhs_col= [x + col for x in[ 0,  1,  1]]
            nbhs_row= [x + row for x in[ -1, 0, -1]] 
        # cell is in the right upper corner
        elif col == ncols -1 and row == 0:
            nbhs_col= [x + col for x in[ -1, -1, 0]]
            nbhs_row= [x + row for x in[  0,  1, 1]]
        # cell is in the right bottom corner
        else:
            nbhs_col= [x + col for x in[ -1, -1, 0 ]]
            nbhs_row= [x + row for x in[ -1,  0, -1]] 
            
    # assuming that a cell in the center has 4 neighbouring cells
    elif four_neighbours == 'True':
        # cell is no edge cell
        if col > 0 and row > 0 and row < nrows -1 and col < ncols -1:
            nbhs_col = [x + col for x in[-1,  0, 0, 1]]
            nbhs_row = [x + row for x in[ 0, -1, 1, 0]]
        # cell is a left edge element but no corner element
        elif col == 0 and row > 0 and row < nrows -1:
            nbhs_col= [x + col for x in[0, 1, 0]]
            nbhs_row= [x + row for x in[-1, 0, 1]]   
        # cell is a right edge element but no corner element
        elif col == ncols -1 and row > 0 and row < nrows -1:
            nbhs_col= [x + col for x in[-1,  0, 0]]
            nbhs_row= [x + row for x in[ 0, 1, -1]]        
        # cell is an upper edge element but no corner element
        elif row == 0 and col > 0 and col < ncols -1:
            nbhs_col= [x + col for x in[-1, 0, 1]]
            nbhs_row= [x + row for x in[ 0, 1, 0]]
        # cell is an bottom edge element but no corner element    
        elif row == nrows -1 and col > 0 and col < ncols -1:
            nbhs_col= [x + col for x in[-1, 0,  1]]
            nbhs_row= [x + row for x in[ 0, -1, 0]] 
        # cell is in the left upper corner
        elif col == 0 and row == 0:
            nbhs_col= [x + col for x in[ 0, 1]]
            nbhs_row= [x + row for x in[ 1, 0]]
        # cell is in the left bottom corner
        elif col == 0 and row == nrows -1:
            nbhs_col= [x + col for x in[ 0,  1]]
            nbhs_row= [x + row for x in[ -1, 0]] 
        # cell is in the right upper corner
        elif col == ncols -1 and row == 0:
            nbhs_col= [x + col for x in[ -1, 0]]
            nbhs_row= [x + row for x in[  0, 1]]
        # cell is in the right bottom corner
        else:
            nbhs_col= [x + col for x in[ -1, 0 ]]
            nbhs_row= [x + row for x in[  0, -1]]

    else:
        raise SystemError("Error: ini input for four_neighbours is not correct")
        req.close_window

    return [nbhs_row, nbhs_col]

def determine_patch_elements(row, col, map, patch_map, patch_ID, cls, four_neighbours):
    """This recursive function scans all patch elements 
       and returns the coordinates of these elements.
       
       input:
           col and row are coordinates of the parent element
           map is the original ascii map
           patch_map is a map with patch_IDs for each patch element
           patch_ID is the ID of the new patch
           cls is the land use index of the patch
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """
    # determine coordinates of neighboring cells
    new_nbhs_row, new_nbhs_col  = getNbh(col, row, map.shape[1], map.shape[0], four_neighbours)
    # stack for patch elements whose neighboring cells should be determined
    nbhs_row = []
    nbhs_col = []
    for i in range(len(new_nbhs_row)):
        # add new neighboring cells to nbhs_row/col if new cells belong to cls and are not jet marked as patch element
        # the cell is no patch element if it has another land use id
        if map[new_nbhs_row[i], new_nbhs_col[i]] == cls and patch_map[new_nbhs_row[i], new_nbhs_col[i]] == 0:
            nbhs_row.append(new_nbhs_row[i])     
            nbhs_col.append(new_nbhs_col[i])  
    while len(nbhs_row) > 0:
        # cells could be double in nbhs_row/col
        if patch_map[nbhs_row[0], nbhs_col[0]] == 0:
            # mark all patch elements in patch_map with patch_ID
            patch_map[nbhs_row[0], nbhs_col[0]] = patch_ID                                            
            # get coordinates of neighboring cells of this cell
            new_nbhs_row, new_nbhs_col  = getNbh(nbhs_col[0], nbhs_row[0], map.shape[1], map.shape[0], four_neighbours)
            for i in range(len(new_nbhs_row)):
                # add new neighboring cells to nbhs_row/col if new cells belong to cls and are not jet marked as patch element
                if map[new_nbhs_row[i], new_nbhs_col[i]] == cls and patch_map[new_nbhs_row[i], new_nbhs_col[i]] == 0:
                    nbhs_row.append(new_nbhs_row[i])     
                    nbhs_col.append(new_nbhs_col[i])
        # delete this checked neighboring cell of the array    
        del nbhs_row[0]
        del nbhs_col[0]

    return patch_map

def create_patch_ID_map(map, NODATA_value, static_elements, four_neighbours):
    """This function clusters the cells of the original map into patches
       and returns a patch ID map as a 2 dimensional array and the start individual as vector.
    
       input: 
           map is the original ascii map
           NODATA_value is the NODATA_value of the original map
           static_elements are the land use indices excluded from the optimization
           four_neighbours if True than 4 neighboring cells are scanned else 8
    """
    
       
    patches= np.zeros([map.shape[0], map.shape[1]], int)
    ids = 0
    NoData = int(NODATA_value)
    genom = []
    # loop over all cells
    for row in range(0, map.shape[0]):
        if (row + 0.0) % (round(map.shape[0] / 10.0)) == 0 :
            progress = ((row+0.0) / map.shape[0]) * 100 
        for col in range(0, map.shape[1]):
            # patchID = 0 used for static_elements
            # map element was not scanned before as patch element and is not a static element or the NODATA_value
            if patches[row,col]==0 and static_elements.count(map[row, col])==0 and map[row,col]!=NoData:
                cls = map[row, col]
                # increment scanned patch ID
                ids += 1
                # marke this cell as scanned patch element 
                patches[row, col] = ids
                determine_patch_elements(row,col, map, patches, ids, cls, four_neighbours) 
                # add the map cell value to the individual vector
                genom.append(cls)
    
                           
    return patches, genom   

patchID_map, genome = create_patch_ID_map(map, header[5], static_elements, four_neighbours)
np.save('processing\\patchID',patchID_map)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

def parse_transition_matrix_from_array(matrix):
    headers = matrix[0][1:]  # Skip -2 in first row
    transition_dict = {}

    for row in matrix[1:]:
        from_class = int(row[0])
        allowed = row[1:]
        transition_dict[from_class] = [int(to) for to, val in zip(headers, allowed) if val == 1]

    return transition_dict

transition_matrix = parse_transition_matrix_from_array(trans_matrix)

def parse_area_constraints(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    headers = [int(x) for x in lines[0].strip().split()[1:]]   # Skip 'land_use'
    min_vals = [int(x) for x in lines[1].strip().split()[1:]]  # Skip 'min'
    max_vals = [int(x) for x in lines[2].strip().split()[1:]]  # Skip 'max'

    constraints = {}
    for i, land_use in enumerate(headers):
        constraints[land_use] = {'min': min_vals[i], 'max': max_vals[i]}

    return constraints

area_constraints = parse_area_constraints(minmax_file)


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def mutate_with_constraints(candidate, mutation_rate, transition_matrix, area_constraints, max_trials=1000):
    """
    Mutate a land-use candidate while respecting transition rules and min/max area constraints.
    
    Args:
        candidate (list): List of land-use class IDs (e.g. [1, 2, 1, 3, 1]).
        mutation_rate (float): Probability of mutating each gene.
        transition_matrix (dict): {class: [allowed_classes_to_switch_to]}.
        area_constraints (dict): {class: {'min': x, 'max': y}} in %.
        max_trials (int): Max attempts to repair if constraints are violated.

    Returns:
        list: Mutated and repaired individual.
    """
    n = len(candidate)
    new_candidate = candidate[:]

    # --- Mutation step
    for i in range(n):
        if random.random() < mutation_rate:
            current_class = new_candidate[i]
            options = transition_matrix.get(current_class, [current_class])
            new_class = random.choice(options)
            new_candidate[i] = new_class

    # --- Repair step to enforce area constraints
    for _ in range(max_trials):
        counts = Counter(new_candidate)
        percentages = {cls: (counts.get(cls, 0) / n) * 100 for cls in area_constraints}

        # Check if within constraints
        violations = []
        for cls, rule in area_constraints.items():
            p = percentages.get(cls, 0)
            if p < rule['min']:
                violations.append((cls, 'min'))
            elif p > rule['max']:
                violations.append((cls, 'max'))

        if not violations:
            return new_candidate  # Valid solution

        # Try to fix one violation at a time
        for cls, kind in violations:
            if kind == 'min':
                # Need more of this class
                for i, val in enumerate(new_candidate):
                    if val != cls and cls in transition_matrix.get(val, []):
                        new_candidate[i] = cls
                        break
            elif kind == 'max':
                # Too much of this class
                for i, val in enumerate(new_candidate):
                    if val == cls:
                        fallback_options = [c for c in transition_matrix.get(cls, []) if c != cls]
                        if fallback_options:
                            new_candidate[i] = random.choice(fallback_options)
                            break
    # If we failed to repair
    print("Warning: Could not fully repair candidate within max trials.")
    return new_candidate



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

class ConstraintAwareMutation(Mutation):

    def __init__(self, transition_matrix, area_constraints, mutation_rate=0.5, max_trials= 1000,**kwargs):
        super().__init__(**kwargs)
        self.transition_matrix = transition_matrix
        self.area_constraints = area_constraints
        self.mutation_rate = mutation_rate
        self.max_trials = max_trials

    def _do(self, problem, X, **kwargs):
        mutated = []

        for candidate in X:
            genome = candidate.tolist()

            # Use your existing mutation function
            new_genome = mutate_with_constraints(
                candidate=genome,
                mutation_rate=self.mutation_rate,
                transition_matrix=self.transition_matrix,
                area_constraints=self.area_constraints,
                max_trials=self.max_trials
            )

            mutated.append(new_genome)

        return np.array(mutated)
    
class MySampling(Sampling):

    def __init__(self, base_genome):
        super().__init__()
        self.base_genome = np.array(base_genome)

    def _do(self,problem, n_samples, **kwargs):
        # Create a population array with shape (n_samples, genome_length)
        genome_length = len(self.base_genome)
        X = np.empty((n_samples, genome_length), dtype=int)

        # First genome is the base genome
        X[0] = self.base_genome

        # Rest are mutations
        for i in range(1, n_samples):
            X[i] = mutate_with_constraints(candidate =self.base_genome, mutation_rate = 0.5 , transition_matrix = transition_matrix, area_constraints = area_constraints)

        return X
    
class ToyModels(ElementwiseProblem):
    def __init__(self,max_yield, max_species):
        super().__init__(n_var=len(genome), n_obj=2, n_constr=4, xl=1, xu=8) #vtype=int
        self.max_yield = max_yield
        self.max_species = max_species


    def _evaluate(self, X, out):
        #print(X)
        X = X.astype(int)
        #print(X)
        # Compute objectives
        f1 = crop_yield(X, soil_fertility_map)
        f2 = forest_species_richness(X)

        # Constraints for max limits
        g1 = f1 - self.max_yield  # Ensures f1 ≤ max_yield
        g2 = f2 - self.max_species  # Ensures f2 ≤ max_species

        # Constraints for min limits
        g3 = 0 - f1  # Ensures f1 ≥ 0
        g4 = 0 - f2  # Ensures f2 ≥ 0
        

        out["F"] = [-f1, -f2]  # Convert to maximization
        out["G"] = [g1, g2, g3, g4]

problem = ToyModels(max_yield=max_yield, max_species=max_species)

#print(type(genome))
genome = np.array(genome)
#print(type(genome))

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
class SimpleDM(AutomatedDM):
    def makeDecision(self, F):
        # Reference point for preference
        ref_x, ref_y = x, y

        # Normalize objectives (used when comparing across different map)
        #F_normalized = F / max_values
        #ref = np.array([ref_x, ref_y]) / max_values
        # Euclidean distances to normalized reference
        #distances = np.linalg.norm(-F_normalized - ref, axis=1)
        


        #Euclidean distances to reference
        distances = np.sqrt((-F[:, 0] - ref_x) ** 2 + (-F[:, 1] - ref_y) ** 2)

        # Decision logic
        if distances[0] < distances[1]:
            return 'a'
        elif distances[1] < distances[0]:
            return 'b'
        else:
            return 'c'
        

simple_dm = SimpleDM()

if type_of_dm == 'manual':
    print('Using Manual DM')
    algorithm = PINSGA2(pop_size=pop_size, tau=dm_period, eta=points, eliminate_duplicates=False,
                         mutation=ConstraintAwareMutation( transition_matrix=transition_matrix, area_constraints=area_constraints, vtype=int), 
                         sampling=MySampling(genome),ranking_type="pairwise")
if type_of_dm == 'automated':
    print('Using Automated DM')
    algorithm = PINSGA2(pop_size=pop_size, tau=dm_period, eta=points, eliminate_duplicates=False,
                         mutation=ConstraintAwareMutation( transition_matrix=transition_matrix, area_constraints=area_constraints, vtype=int), 
                         sampling=MySampling(genome),ranking_type="pairwise", automated_dm=simple_dm)
else:
    print('Choose a type of DM')

start_time = time.time()
print_timestamp("PINSGA2 Optimization Started")

res_pinsga = minimize(problem, algorithm, termination=('n_gen', max_gen), seed=seeds, save_history=False, verbose=True)

print_timestamp("PINSGA2 Optimization Finished")

end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plot = Scatter(title="PINSGA-II")
plot.add(-res_pinsga.F, label="PINSGA-II")

plot.do()
plot.ax.set_xlim(0, max_yield)
plot.ax.set_ylim(0, max_species)
plot.ax.set_xlabel("Crop Yield")
plot.ax.set_ylabel("Forest Species Richness")
plot.fig.savefig(f"output/pinsga2_plot{map_size}_{timestamp}.png", dpi=300)
plot.show()

#df = pd.DataFrame(-res_pinsga.F, columns=['Crop Yield', 'Forest Species Richness'])
#df.to_csv('pinsga2.csv', index=False)

objectives = -res_pinsga.F

# Convert genomes to DataFrame
genomes = pd.DataFrame(res_pinsga.X.astype(int))  # Each row is a genome (land use classes)

# Combine objectives and genomes
df = pd.DataFrame(objectives, columns=['Crop Yield', 'Forest Species Richness'])
df_full = pd.concat([df, genomes], axis=1)

# Save to CSV
#df_full.to_csv('output/pinsga2.csv', index=False)
df_full.to_csv(f'output/pinsga2{map_size}_{timestamp}.csv', index=False)