#holy grail numpy ;-)
import numpy as np

#other dependencies
from collections import defaultdict
from typing import List, Tuple, Union

#scipy to ease things up a bit
import scipy as sp



#function to find the cosine similarity between two vectors

def cosine_similarity(v1 : np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)

    cosine = dot / (v1_mag * v2_mag)

    return cosine

#function to find the euclidean distance between two vectors

def euclidean_similarity(v1 : np.ndarray, v2: np.ndarray) -> float:
    sub = v1 - v2
    return np.sqrt(np.dot(sub.T, sub))


#function to calculate hamming distance

def hamming_similarity (v1 : np.ndarray, v2: np.ndarray) -> float:
    return (sp.spatial.distance.hamming(v1, v2)* v1.size)

#creating the main class

class VDB:

    #builder
    def __init__(self):
        self.vectors = defaultdict(np.ndarray)

    #insert vector method

    def insert(self, key: str, vector:np.ndarray) -> None:
        self.vectors[key] = vector

    #Similarity search method

    def search(self, query_vec:np.ndarray, k: int , metric:Union['cosine', 'euclidean', 'hamming']) -> List[Tuple[str,float]]:
        if metric not in ['cosine', 'euclidean', 'hamming']:
            raise ValueError("Invalid Metric Selection")
        else:
            if metric == 'cosine':
                sims = [(key, cosine_similarity(query_vec, vector)) for key, vector in self.vectors.items()]
                sims.sort(key = lambda x: x[1], reverse= True)
                return sims[:k]
            elif metric == 'euclidean':
                sims = [(key, euclidean_similarity(query_vec, vector)) for key, vector in self.vectors.items()]
                sims.sort(key = lambda x: x[1], reverse= True)
                return sims[:k]
            else:
                sims = [(key, hamming_similarity(query_vec, vector)) for key, vector in self.vectors.items()]
                sims.sort(key = lambda x: x[1], reverse= True)
                return sims[:k]  
    #query result method
    def query_result(self, key: str) -> np.ndarray:
        return self.vectors.get(key, None) 
    





# Create an instance of the VectorDatabase
vector_db = VDB()

# Insert vectors into the database
vector_db.insert("vector_1", np.array([0.1, 0.2, 0.3]))
vector_db.insert("vector_2", np.array([0.4, 0.5, 0.6]))
vector_db.insert("vector_3", np.array([0.7, 0.8, 0.9]))

# Search for similar vectors
query_vector = np.array([0.15, 0.25, 0.35])
similar_vectors = vector_db.search(query_vector, k=2, metric = 'hamming')
print("Similar vectors:", similar_vectors)

# Retrieve a specific vector by its key
retrieved_vector = vector_db.query_result("vector_1")
print("Retrieved vector:", retrieved_vector)





