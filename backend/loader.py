import gzip
import pickle

def load_compressed_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
