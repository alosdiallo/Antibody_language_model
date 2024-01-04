import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import resource
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Configure the logging module
logging.basicConfig(filename='language_model_DNA.log', level=logging.INFO)

# Set up a logger instance
logger = logging.getLogger()

logger.info("Loading Model")
# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

model = model.to(device)
logger.info("Done")
# If there's more than one GPU available, wrap the model with DataParallel.
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

resource.setrlimit(resource.RLIMIT_NOFILE, (3096, 3096))


def process_sequences(dna_sequences):
    dna_sequences = [seq if len(seq) % 3 == 0 else seq[:-(len(seq) % 3)] for seq in dna_sequences]
    tokens = tokenizer.batch_encode_plus(dna_sequences, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens.to(device))
    pooled_outputs = outputs[0]  # Use the prediction scores instead
    pooled_outputs = pooled_outputs.mean(dim=1)  # Average the embeddings across the sequence length
    
    # Normalize embeddings
    pooled_outputs = (pooled_outputs - pooled_outputs.mean()) / pooled_outputs.std()
    
    return pooled_outputs.cpu().numpy()


chunksize = 6000
data = pd.read_csv("FDA045_Data_Small.csv", chunksize=chunksize)

embeddings = []
umis = []
shm_values = []
logger.info("Work on embeddings")
for chunk in data:
    sequences_chunk = chunk["IgH Nucleotide Sequence"].tolist()
    umis_chunk = chunk["UMI"].tolist()
    shm_values_chunk = chunk["IgH SHM"].tolist()

    embeddings_chunk = process_sequences(sequences_chunk)

    embeddings.extend(embeddings_chunk)
    umis.extend(umis_chunk)
    shm_values.extend(shm_values_chunk)
logger.info("Done")
np.save('embeddings_gpu.npy', embeddings)
