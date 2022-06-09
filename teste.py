from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import pickle

embeddings = np.load("embeddings.npy")
embeddings_sparse = sparse.csr_matrix(embeddings)
cos_mat = cosine_similarity(embeddings_sparse, dense_output=False)
print(cos_mat.shape[0])
avgs = []
for i in range(cos_mat.shape[0]):
    avg = np.mean(cos_mat[i])
    avgs.append(avg)
np.save("avg_cos_mat.dat", avgs)
#with open('cos_mat_data.dat', 'wb') as outfile:
#	pickle.dump(cos_mat, outfile, pickle.HIGHEST_PROTOCOL)