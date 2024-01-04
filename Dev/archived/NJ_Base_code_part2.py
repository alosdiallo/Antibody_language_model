import pandas as pd
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
from multiprocessing.pool import ThreadPool
import logging
import gc
from ete3 import Tree
from io import StringIO

# Configure the logging module
logging.basicConfig(filename='language_model.log', level=logging.INFO)

# Set up a logger instance
logger = logging.getLogger()

logger.info("Loading modified distance matrix from 'modified_distance_matrix.csv'")
df = pd.read_csv('modified_distance_matrix.csv', index_col=0)

modified_distance_matrix = df.values.tolist()
umis = list(df.columns)

def construct_partial_tree(submatrix, labels):
    dist_matrix = DistanceMatrix(list(labels), submatrix)
    constructor = DistanceTreeConstructor()
    partial_tree = constructor.nj(dist_matrix)

    del submatrix, labels, dist_matrix, constructor
    gc.collect()

    return partial_tree

logger.info("Finished with the partial tree")


num_chunks = 20
chunk_size = len(umis) // num_chunks
chunks = []
chunk_labels = []

for i in range(0, len(umis), chunk_size):
    end_index = i + chunk_size if i + chunk_size <= len(umis) else len(umis)
    submatrix = [modified_distance_matrix[j][:j+1-i] for j in range(i, end_index)]
    sublabels = umis[i:end_index]
    chunks.append(submatrix)
    chunk_labels.append(sublabels)

logger.info("Finished with the last nested loop")
# Set up multiprocessing pool with the desired number of processes
num_processes = 20
pool = ThreadPool(num_processes)

# Construct partial trees in parallel for each chunk of the distance matrix
partial_trees = pool.starmap(construct_partial_tree, zip(chunks, chunk_labels))
logger.info("Construct partial trees in parallel for each chunk of the distance matrix")
# Close the multiprocessing pool
pool.close()
logger.info("Multicore work done")
logger.info("About to merge")

del chunks, chunk_labels
gc.collect()

# Merge the partial trees into the final Neighbor-Joining tree
def merge_trees(tree1, tree2):
    # Convert Biopython Tree objects to ETE3 Tree objects
    newick_tree1 = tree1.format('newick').strip()
    newick_tree2 = tree2.format('newick').strip()

    ete_tree1 = Tree(newick_tree1, format=1)
    ete_tree2 = Tree(newick_tree2, format=1)

    # Merge the trees
    ete_tree1.add_child(ete_tree2)

    # Convert the merged ETE3 Tree object back to a Biopython Tree object
    merged_tree = Phylo.read(StringIO(ete_tree1.write(format=1)), 'newick')
    return merged_tree

# Merge the partial trees into the final Neighbor-Joining tree
final_tree = partial_trees[0]
for i in range(1, num_chunks):
    final_tree = merge_trees(final_tree, partial_trees[i])
	
logger.info("Finished with tree")

# Save the tree to a Newick format file
Phylo.write(final_tree, 'NJ_Tree.newick', 'newick')

print("Neighbor-Joining tree has been saved to 'NJ_Tree.newick'")
