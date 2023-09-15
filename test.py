import os
import pandas as pd
import numpy as np
import warnings
import scipy as sp 
pd.options.display.max_columns

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
anime_data = pd.read_csv('C:\\Users\\udit hasija\\Downloads\\archive (1)\\anime.csv')
rating_data = pd.read_csv('C:\\Users\\udit hasija\\Downloads\\archive (1)\\rating.csv')
anime_data = anime_data[~np.isnan(anime_data['rating'])]
anime_data['genre'] = anime_data['genre'].fillna(anime_data['genre'].dropna().mode().values[0])
anime_data['type'] = anime_data['type'].fillna(anime_data['type'].dropna().mode().values[0])
rating_data['rating'] = rating_data['rating'].apply(lambda x: np.nan if x==-1 else x)
atype = 'TV'
anime_data = anime_data[anime_data['type']==atype]
anime_full = pd.merge(anime_data , rating_data , on = 'anime_id')
anime_full = anime_full.rename(columns= {'rating_x': 'rating' , 'rating_y':'user_rating'})
anime_full = anime_full[['user_id', 'name', 'user_rating', 'Synopsis']]
anime_7500= anime_full[anime_full.user_id <= 25000]
pivot = anime_7500.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
pivot_n.fillna(0 , inplace= True)
pivot_n = pivot_n.T
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)
anime_similarity = cosine_similarity(piv_sparse)
ani_sim_df = pd.DataFrame(anime_similarity, index = pivot_n.index, columns = pivot_n.index)
recommendations = {'name': [], 'synopsis': []}
def anime_recommendation(ani_name):
        number = 1
        for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6]:
            synop = anime_data[anime_data['name'] == anime]
            synopsis = synop['Synopsis'].values[0]
            recommendations['name'].append(anime)
            recommendations['synopsis'].append(synopsis)
            number +=1  
        print(recommendations)

anime_recommendation('Naruto')