import os
import numpy as np
import scipy.sparse as sps

def load_data_into_matrix(filename):
    data = np.load(filename)
    cutoff = 100000
    data = sps.coo_matrix((data[:cutoff, 2], 
                           (data[:cutoff, 0] - 1, 
                            data[:cutoff, 1] - 1)), 
                          shape=(103703, 17770),
                          dtype=np.int8)
    return data

def load_random_data_into_matrix(n_u, n_m):
    matrix = np.random.randint(0, 5, (n_u, n_m))
    return matrix

def cosine_sim(u, v):
    return 1 - np.arccos(np.dot(u, v) / np.sqrt(np.sum(np.square(u)) * np.sum(np.square(v)))) / np.pi

np.random.seed(42)

username = os.getcwd().split('/')[2]
username = ''
print('username:', username)

if username == 'mvgroeningen':
    n_users = 100
    n_movies = 50
    rating_matrix = load_random_data_into_matrix(n_users, n_movies) 
else:
    filename = 'user_movie_rating.npy'
    rating_matrix = load_data_into_matrix(filename)

print(rating_matrix.getrow(0).toarray()[0])
print(cosine_sim(rating_matrix.getrow(0).toarray()[0], rating_matrix.getrow(8).toarray()[0]))
