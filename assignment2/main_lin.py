import time
import argparse
import numpy as np
import scipy.sparse as sps
from itertools import repeat


def load_data_into_matrix(filename, n_users, n_movies, n_ratings, discrete=False):
    """
    Create a sparse row matrix from the discretized ratings data.
    
    filename: string, name of the file containing the data
    n_users: int, number of users
    n_movies: int, number of movies
    n_ratings: int, number of ratings
    discrete: bool, whether to use discrete data
    """
    data = np.load(filename)
    if discrete:
        ratings = np.ones(n_ratings)
    else:
        ratings = data[:, 2]
    users = data[:, 0] - 1
    movies = data[:, 1] - 1
    csr = sps.csr_matrix((ratings, (users, movies)), shape=(n_users, n_movies), dtype=np.int32)
    return csr


def sign_js_sim(signatures, user_pair):
    """
    Calculate the Jaccard similarity between the signatures of two users.
    signatures: array, matrix with a number of signatures for each user
    u: int, user index
    """
    return np.mean(signatures.T[user_pair[0]] == signatures.T[user_pair[1]])
    
    
def true_js_sim(sps_rating_matrix, user_pair):
    """
    Calculate the Jaccard similarity between the movie ratings of two users.
    signatures: array, matrix with a number of signatures for each user
    user_pair: array, array containing the indices of two users
    """
    intersection = (sps_rating_matrix[user_pair[0]].multiply(sps_rating_matrix[user_pair[1]])).count_nonzero()
    union = (sps_rating_matrix[user_pair[0]] + sps_rating_matrix[user_pair[1]]).count_nonzero()
    sim = intersection / union
    return sim


def cosine_sim(u, v):
    """
    Calculate the cosine similarity between the movie ratings of two users.
    
    u: array, movie ratings of a user
    v: array, movie ratings of another user
    """
    p1_p2 = np.dot(u, v)
    norm_p1_norm_p2 = np.sqrt(np.sum(np.square(u)) * np.sum(np.square(v)))
    if norm_p1_norm_p2 == 0:
        return 0.5
    else:
        return 1 - np.arccos(p1_p2 / norm_p1_norm_p2) / np.pi
        
        
def unique_pairs_from_array(arr):
    """
    Finds all unique pairs in an array that does not contain duplicates.
    
    arr: array, array that does not contain duplicate values
    """
    pairs = []
    n = len(arr)
    for idx, val in enumerate(arr[:-1]):
        pairs_for_single_value = np.stack((np.tile(val, n - idx - 1), arr[idx + 1:]), axis=1)
        pairs.append(pairs_for_single_value)
    return np.concatenate(pairs, axis=0)


def js_signatures(sps_rating_matrix, n_permutations):
    """
    The first method used for generating the signature matrix. We read 
    the ratings for user[j] following the permutation[i] order using the for loop
    and take the index of the first rating we meet as the signature[i][j]
    """
    
    n_users = sps_rating_matrix.shape[0]
    n_movies = sps_rating_matrix.shape[1]
    permutations = np.array([np.random.permutation(n_movies) for i in range(n_permutations)])
    signatures = np.full((n_permutations, n_users), np.inf)
    for p in range(n_permutations):
        # get the indices that would sort the permutation arrays in order to access the ratings following
        # the permutation order.
        permutation_matrix = sps_rating_matrix[:, permutations[p]]
        for u in range(n_users):
            signatures[p][u] = permutation_matrix.indices[permutation_matrix.indptr[u]:permutation_matrix.indptr[u + 1]].min()

    return signatures


def make_random_projections(n_projections, n_movies):
    """
    Make a sparse matrix containing random projections in N_movies-dimensional space.
    
    n_projections: int, number of projections
    n_movies: int, number of movies
    """
    flat_v = np.random.choice([-1, 1], size=(n_projections * n_movies))
    mesh = np.array(np.meshgrid(np.arange(n_projections), np.arange(n_movies))).T.reshape(-1,2)
    projection_indices, movie_indices = mesh[:, 0], mesh[:, 1]
    return sps.csr_matrix((flat_v, (projection_indices, movie_indices)), 
                          shape=(n_projections, n_movies), dtype=np.float32)
                          
                          
def cs_signatures(sps_rating_matrix, n_projections):
    """
    Create signatures for each user by taking the dot product with random projections.
    
    rating_matrix: scipy.sparse.csr_matrix, sparse row matrix containing the ratings
    n_projections: int, the number of projections vectors to use for the signatures
    """

    # Create a number of random projection vectors in N_movie-dimensional space
    n_movies = sps_rating_matrix.shape[1]
    v = make_random_projections(n_projections, n_movies)

    # Make a signature matrix by taking the dot product of the rating matrix with the random projections matrix
    # Convert all positive values to 1 and all negative values to 0 
    # [n_projections, n_movies] @ [n_movies, n_users] -> [n_projections, n_users]
    signatures = ((v * sps_rating_matrix.T).toarray() > 0).astype(np.int32)
    
    return signatures


def find_js_pairs(sps_rating_matrix, signatures, b, permutations_per_band, threshold):
    """
    Find similar user pairs in a sparse rating matrix with the Jaccard similarity method.
    sps_rating_matrix: scipy.sparse.csr_matrix, sparse row matrix where each row contains
    the movie ratings of a user
    signatures: array, matrix with a number of signatures for each user
    b: int, band index
    projections_per_band: int, the number of projections in each band
    threshold: float, should be a value between 0 and 1.
    """
    jaccard_pairs = np.array([])
    
    start_of_band = b * permutations_per_band
    end_of_band = (b + 1) * permutations_per_band
    
    band = signatures[start_of_band:end_of_band, :]
    key, inds, invs = np.unique(band, axis=1, return_index=True, return_inverse=True)
    
    for i in range(len(key.T)):
        bucket = np.where(invs == i)[0]
        if len(bucket) > 1:
            candidate_pairs = unique_pairs_from_array(bucket)
            
            signature_similarities = []
            for pair in candidate_pairs:
                signature_similarities.append(sign_js_sim(signatures, pair))
            
            signature_pairs = candidate_pairs[np.array(signature_similarities) > threshold]
            
            jaccard_similarities = []
            for pair in signature_pairs:
                jaccard_similarities.append(true_js_sim(sps_rating_matrix, pair))
            
            jaccard_similarities = np.array(jaccard_similarities)
            if len(jaccard_similarities[jaccard_similarities > threshold] > 0):
                if len(jaccard_pairs) != 0:
                    jaccard_pairs = np.concatenate((jaccard_pairs, signature_pairs[jaccard_similarities > threshold]), axis=0)
                else:
                    jaccard_pairs = signature_pairs[jaccard_similarities > threshold]
        
    return jaccard_pairs


def find_candidate_pairs(signatures, n_bands, projections_per_band):
    """
    Find candidate pairs that have the same signatures in a certain band. Employs the LSH algorithm.
    
    signatures: array, matrix containing signatures for each user with shape [n_projections, n_users]
    n_bands: int, the number of bands to use for LSH
    projections_per_band: int, the number of projections in each band
    """
    n_projections = n_bands * projections_per_band

    binary_array = np.array([2**i for i in np.arange(projections_per_band)[::-1]])
    candidate_pairs = []

    for b in range(n_bands):
        # Select a band from the signatures
        start_of_band = b * projections_per_band
        end_of_band = min(start_of_band + projections_per_band, n_projections)
        band = signatures[start_of_band:end_of_band, :]

        # The signatures of each user make up an array of 1s and 0s. We can treat this as a binary number 
        # to give each user an integer value as its signature.
        band = np.dot(binary_array, band).T

        # Find all unique signatures in the band
        unique_signatures = np.unique(band)

        # For each unique signature, find the indices where that signature occurs
        for u in unique_signatures:
            indices_of_unique_signature = np.where(band == u)[0]

            # If more than 1 index is found, it means that multiple users have the same signatures
            # in the band and are likely to be similar
            if indices_of_unique_signature.shape[0] > 1:
                pairs_with_unique_signature = unique_pairs_from_array(indices_of_unique_signature)
                candidate_pairs.append(pairs_with_unique_signature)

    # Concatenate the candidate pairs found for all bands and unique signatures
    # and remove duplicate pairs
    if len(candidate_pairs) > 0:
        candidate_pairs = np.unique(np.concatenate(candidate_pairs, axis=0), axis=0).astype(np.int32)
    else:
        candidate_pairs = np.array(candidate_pairs).astype(np.int32)
        print('No pairs found!')
    
    return candidate_pairs


def find_signature_pairs(candidate_pairs, signatures, threshold=0.73):
    """
    Find the pairs with a signature similarity above a certain threshold from an array of candidate pairs.
    
    candidate_pairs: array, array containing the user indices of candidate pairs with shape [n_pairs, 2]
    signatures: array, matrix containing signatures for each user with shape [n_projections, n_users] 
    threshold: float, if the fraction of identical signatures between two users is above this treshold,
    then they are selected as a signature pair. Should be a value between 0 and 1.
    """
    
    n_projections = signatures.shape[0]
    
    signature_similarities = []
    signatures = np.where(signatures == 0, -1, signatures)
    for user_1, user_2 in candidate_pairs:
        signature_similarities.append(np.dot(signatures[:, user_1], signatures[:, user_2]))
        
    sig_threshold = n_projections * (2 * threshold - 1)
    signature_pairs = candidate_pairs[np.array(signature_similarities) > sig_threshold]
    
    return signature_pairs

    
def find_cosine_pairs(signature_pairs, csr, threshold=0.73):
    """
    Find the pairs with a cosine similarity above a certain threshold from an array of signature pairs.
    
    csr: scipy.sparse.csr_matrix, sparse row matrix where each row contains the movie ratings of a user
    candidate_pairs: array, array containing the user indices of candidate pairs with shape [n_pairs, 2]
    threshold: float, if the cosine similarity of two users is above this threshold, then they are 
    selected as a cosine pair. Should be a value between 0 and 1.
    """
    
    cosine_similarities = []
    for user_1, user_2 in signature_pairs:
        u1_ratings = csr[user_1].toarray()[0]
        u2_ratings = csr[user_2].toarray()[0]
        cosine_similarities.append(cosine_sim(u1_ratings, u2_ratings))

    cosine_pairs = signature_pairs[np.array(cosine_similarities) > threshold]
    
    return cosine_pairs


def js_main(sps_rating_matrix, n_bands, permutations_per_band, threshold):
    """
    Main function for running the Jaccard Similarity method
    sps_rating_matrix: scipy.sparse.csr_matrix, sparse row matrix where each row contains
    n_bands: int, the number of bands to use for LSH
    permutations_per_band: int, the number of permutations in each band
    threshold: float, should be a value between 0 and 1.
    """

    print('number of bands:', n_bands)
    print('number of permutations per band:', permutations_per_band)

    n_permutations = n_bands * permutations_per_band
    signatures = js_signatures(sps_rating_matrix, n_permutations)
    
    pair_list = []
    for band in range(n_bands):
        pairs = find_js_pairs(sps_rating_matrix, signatures, band, permutations_per_band, threshold)
        if len(pairs) > 0:
            pair_list.append(pairs)
    pair_list = np.unique(np.concatenate(pair_list, axis=0), axis=0)
    
    print(f'Number of pairs with Jaccard sim > {threshold}:', len(pair_list))
    
    return pair_list


def cs_main(sps_rating_matrix, n_bands, projections_per_band, threshold):
    """
    Find similar user pairs in a sparse rating matrix. This is done in 4 steps.
    
    Step 1: Calculate a number of signatures for each user
    Step 2: Find candidate pairs by finding users that have the same signatures in a certain band
    Step 3: Filter the candidate pairs by taking the pairs where the fraction of similar signatures
    is above a certain threshold
    Step 4: Filter the remaining pairs by taking the pairs that have a cosine similarity above a 
    certain threshold.
    
    sps_rating_matrix: scipy.sparse.csr_matrix, sparse row matrix where each row contains 
    the movie ratings of a user
    n_bands: int, the number of bands to use for LSH
    projections_per_band: int, the number of projections in each band
    threshold: float, should be a value between 0 and 1.
    """
    
    print('number of bands:', n_bands)
    print('number of projections per band:', projections_per_band)

    n_projections = n_bands * projections_per_band

    signatures = cs_signatures(sps_rating_matrix, n_projections)
    
    candidate_pairs = find_candidate_pairs(signatures, n_bands, projections_per_band)
    signature_pairs = find_signature_pairs(candidate_pairs, signatures, threshold)
    cosine_pairs = find_cosine_pairs(signature_pairs, sps_rating_matrix, threshold)

    n_cosine_pairs = cosine_pairs.shape[0]
    print(f'Number of pairs with cos_sim > {threshold}:', n_cosine_pairs)
    
    return cosine_pairs


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help="Data file path")
parser.add_argument('-s', '--seed', type=int, help="Random seed")
parser.add_argument('-m', '--measure', type=str, help="Similarity measure (js/cs/dcs)")


if __name__ == '__main__':

    args = parser.parse_args()
    np.random.seed(args.seed)
    
    n_users = 103703
    n_movies = 17770
    n_ratings = 65225506
    filename = args.data

    begin_run = time.time()

    if args.measure == 'js':
        n_bands = 20
        permutations_per_band = 5
        threshold = 0.50

        csr = load_data_into_matrix(filename, n_users, n_movies, n_ratings, discrete=False)
        similar_users = js_main(csr, n_bands, permutations_per_band, threshold)
        output_file = 'js.txt'

    elif args.measure == 'cs':
        n_bands = 10
        projections_per_band = 15
        threshold = 0.73

        csr = load_data_into_matrix(filename, n_users, n_movies, n_ratings, discrete=False)
        similar_users = cs_main(csr, n_bands, projections_per_band, threshold)
        output_file = 'cs.txt'

    elif args.measure == 'dcs':
        n_bands = 10
        projections_per_band = 15
        threshold = 0.73

        csr = load_data_into_matrix(filename, n_users, n_movies, n_ratings, discrete=True)
        similar_users = cs_main(csr, n_bands, projections_per_band, threshold)
        output_file = 'dcs.txt'

    else:
        raise ValueError(f'Similarity measure unknown, {args.measure} is not in js/cs/dcs.')

    run_time = time.time() - begin_run
    print('Run time', int(run_time // 60), 'min', int(run_time % 60), 'sec')

    np.savetxt(output_file, similar_users, fmt='%i', delimiter=',') 
    
