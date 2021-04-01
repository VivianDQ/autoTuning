import numpy as np
import pickle
import rpy2.robjects as ro

def mf(input_path, output_path, d, dataname):
    # path: path to data
    # output_path needs to end with "/"
    # dim: a str denoting the dimension of matrices
    r=ro.r
    r.source("data/mf.R")
    r.mf(input_path, output_path, d, dataname)

def process_netflix_data():
    lines = []
    movie_count_rate = {}
    count = 0
    item = None
    for i in range(1,5):
        with open('data/raw_data/combined_data_{}.txt'.format(i), 'r') as f: 
            for line in f:
                if ":" in line:
                    if item != None:
                        movie_count_rate[int(item)] = count
                    item = line.split(":")[0]
                    count = 0
                    continue  
                user, rating = line.split(',')[:2]
                count +=1
                lines += [[int(user), int(item), float(rating)]]
    print('data reading done')

    np.savetxt('data/raw_data/netflix_ratings.txt', lines)
    print('ratings saving done')

    inpath = 'data/raw_data/netflix_ratings.txt'
    outpath = 'data/'
    dim = '10'
    mf(inpath, outpath, dim, 'netflix_')

    users = np.loadtxt("data/netflix_users_matrix_d{}".format(dim))
    movies = np.loadtxt("data/netflix_movies_matrix_d{}".format(dim))
    
    idx = []
    for k,c in movie_count_rate.items():
        if c>=10000: # only use those movies that have more than 10000 ratings
            idx.append(int(k))
    
    users = users[np.any(users != 0, axis=1)]
    movies = movies[idx,:]
    np.savetxt("data/netflix_users_matrix_d{}".format(dim), users)
    np.savetxt("data/netflix_movies_matrix_d{}".format(dim), movies)
    print('mf processing done, data saved')
    
def process_movielens_data():
    lines = []
    with open('data/raw_data/u.data', 'r') as f: # u.data is from the movielens100K data
        for line in f:
            user, item, rating = line.split('\t')[:3]
            lines += [[int(user), int(item), float(rating)]]
    print('data reading done')
    
    # save user_idx, movie_idx, ratings data
    np.savetxt('data/raw_data/movielens100k_ratings.txt', lines)
    print('ratings saving done')
    # 100,000 ratings (1-5) from 943 users on 1682 movies. 

    inpath = 'data/raw_data/movielens100k_ratings.txt'
    outpath = 'data/'
    dim = '10'
    mf(inpath, outpath, dim, 'movielens_')
    print('mf processing done, data saved')