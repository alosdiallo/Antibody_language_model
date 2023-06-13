import pandas as pd
import torch
import numpy as np
from tape import ProteinBertModel, TAPETokenizer
from Bio.Seq import Seq
from sklearn.metrics.pairwise import cosine_similarity
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
from multiprocessing.pool import ThreadPool
from functools import partial
import logging
import resource
import gc

resource.setrlimit(resource.RLIMIT_NOFILE, (2096, 2096))

# Initialize the tokenizer and model
tokenizer = TAPETokenizer(vocab='iupac')
model = ProteinBertModel.from_pretrained('bert-base')

# Configure the logging module
logging.basicConfig(filename='language_model.log', level=logging.INFO)

# Set up a logger instance
logger = logging.getLogger()

logger.info("Process Sequence")
def process_sequence(dna_sequence, tokenizer, model):
    if len(dna_sequence) % 3 != 0:
        dna_sequence = dna_sequence[:-(len(dna_sequence) % 3)]

    seq = Seq(dna_sequence)
    aa_sequence = seq.translate()[:-1]
    aa_sequence = str(aa_sequence).replace('*', '')

    tokens = torch.tensor([tokenizer.encode(aa_sequence)])
    with torch.no_grad():
        outputs = model(tokens)
        pooled_output = outputs[1]

    return pooled_output.numpy().flatten()

chunksize = 500
data = pd.read_csv("FDA045_Data_Small.csv", chunksize=chunksize)

embeddings = []
umis = []
shm_values = []
logger.info("Finnished process sequence")

for chunk in data:
    sequences_chunk = chunk["IgH Nucleotide Sequence"].tolist()
    umis_chunk = chunk["UMI"].tolist()
    shm_values_chunk = chunk["IgH SHM"].tolist()

    num_processes = 6
    pool = ThreadPool(num_processes)
    process_sequence_partial = partial(process_sequence, tokenizer=tokenizer, model=model)
    embeddings_chunk = pool.map(process_sequence_partial, sequences_chunk)

    embeddings.extend(embeddings_chunk)
    umis.extend(umis_chunk)
    shm_values.extend(shm_values_chunk)

logger.info("Computing pairwise cosine similarity between embeddings")
similarity_matrix = cosine_similarity(embeddings)

logger.info("Saving similarity matrix to 'Sequence_Similarity_Matrix.csv'")
similarity_df = pd.DataFrame(similarity_matrix, columns=umis, index=umis)
similarity_df.to_csv("Sequence_Similarity_Matrix.csv")

logger.info("Converting cosine similarity to distance")
distance_matrix = 1 - similarity_matrix

logger.info("Incorporating somatic hypermutation rates into the distance matrix")
num_sequences = len(shm_values)
modified_distance_matrix = []
del similarity_matrix, similarity_df
gc.collect()

for i in range(num_sequences):
    row = []
    for j in range(i + 1):
        avg_shm = (shm_values[i] + shm_values[j]) / 2
        modified_distance = distance_matrix[i, j] * avg_shm
        row.append(modified_distance)
    modified_distance_matrix.append(row)

df = pd.DataFrame(modified_distance_matrix, columns=umis, index=umis)

logger.info("Saving modified distance matrix to 'modified_distance_matrix.csv'")
df.to_csv('modified_distance_matrix.csv')

del distance_matrix, df
gc.collect()
