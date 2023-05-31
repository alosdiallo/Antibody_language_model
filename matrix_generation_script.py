import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import gc

# Configure the logging module
logging.basicConfig(filename='Cosine_language_model_DNA.log', level=logging.INFO)

# Set up a logger instance
logger = logging.getLogger()

logger.info("Loading Data")
# Read the data
data = pd.read_csv("FDA045_Data_Small.csv")

# Recreate umis and shm_values
umis = data["UMI"].tolist()
shm_values = data["IgH SHM"].tolist()

# Load the embeddings from the file
embeddings = np.load('embeddings_gpu.npy')
logger.info("Done")

def batch_cosine_similarity(X, Y=None, batch_size=1000):
    if Y is None:
        Y = X

    similarity_matrix = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(0, X.shape[0], batch_size):
        for j in range(0, Y.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[j:j+batch_size]
            similarity_matrix[i:i+batch_size, j:j+batch_size] = cosine_similarity(X_batch, Y_batch)

    return similarity_matrix

logger.info("Compute the similarity matrix in batches")
# Compute the similarity matrix in batches
similarity_matrix = batch_cosine_similarity(np.array(embeddings))
logger.info("Done")

logger.info("Saving similarity matrix to 'Sequence_Similarity_Matrix.csv'")
similarity_df = pd.DataFrame(similarity_matrix, columns=umis, index=umis)
similarity_df.to_csv("Sequence_Similarity_Matrix_DNA.csv")

logger.info("Converting cosine similarity to distance")
distance_matrix = 1 - similarity_matrix

logger.info("Incorporating somatic hypermutation rates into the distance matrix")
num_sequences = len(shm_values)
modified_distance_matrix = []
del similarity_matrix, similarity_df
gc.collect()

logger.info("Loop start")
for i in range(num_sequences):
    row = []
    for j in range(num_sequences):  # Notice the change here
        avg_shm = (shm_values[i] + shm_values[j]) / 2
        modified_distance = distance_matrix[i, j] * avg_shm
        row.append(modified_distance)
    modified_distance_matrix.append(row)
logger.info("Finished SHM")

df = pd.DataFrame(modified_distance_matrix, columns=umis, index=umis)

logger.info("Saving modified distance matrix to 'modified_distance_matrix.csv'")
df.to_csv('modified_distance_matrix_DNA.csv')

del distance_matrix, df
gc.collect()
