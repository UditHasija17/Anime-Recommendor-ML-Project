from flask import Flask, render_template, request
import pandas as pd
anime_data = pd.read_csv('C:\\Users\\udit hasija\\Documents\\Project Anime\\data\\anime.csv')
anilist = anime_data['name']
def get_anime_recommendations(anime_type, anime_name, num_recommendations):
    import os
    import pandas as pd
    import numpy as np
    import warnings
    import scipy as sp 
    pd.options.display.max_columns

    warnings.filterwarnings("always")
    warnings.filterwarnings("ignore")

    from sklearn.metrics.pairwise import cosine_similarity
    anime_data = pd.read_csv('C:\\Users\\udit hasija\\Documents\\Project Anime\\data\\anime.csv')
    rating_data = pd.read_csv('C:\\Users\\udit hasija\\Documents\\Project Anime\\data\\rating.csv')
    anime_data = anime_data[~np.isnan(anime_data['rating'])]
    anime_data['genre'] = anime_data['genre'].fillna(anime_data['genre'].dropna().mode().values[0])
    anime_data['type'] = anime_data['type'].fillna(anime_data['type'].dropna().mode().values[0])
    rating_data['rating'] = rating_data['rating'].apply(lambda x: np.nan if x==-1 else x)
    atype = anime_type
    anime_data = anime_data[anime_data['type']==atype]
    anime_full = pd.merge(anime_data , rating_data , on = 'anime_id')
    anime_full = anime_full.rename(columns= {'rating_x': 'rating' , 'rating_y':'user_rating'})
    anime_full = anime_full[['user_id', 'name', 'user_rating']]
    anime_7500= anime_full[anime_full.user_id <= 25000]
    pivot = anime_7500.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
    pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    pivot_n.fillna(0 , inplace= True)
    pivot_n = pivot_n.T
    pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]
    pivsparse = sp.sparse.csr_matrix(pivot_n.values)
    similarity = cosine_similarity(pivsparse)
    ani_df = pd.DataFrame(similarity, index = pivot_n.index, columns = pivot_n.index)
    recommendations = {'name': [], 'match' : [] ,'synopsis': [] ,'episodes':[]}
    number = 1
    for anime in ani_df.sort_values(by = anime_name, ascending = False).index[1:int(num_recommendations)+1]:
        synop = anime_data[anime_data['name'] == anime]
        synopsis = synop['Synopsis'].values[0]
        perc = round(ani_df[anime][anime_name]*100,2)
        epi = synop['episodes'].values[0]
        recommendations['name'].append(anime)
        recommendations['synopsis'].append(synopsis)
        recommendations['episodes'].append(int(epi))
        recommendations['match'].append(perc)
        number +=1  
    return recommendations

app = Flask(__name__)
app._static_folder = 'C:\\Users\\udit hasija\\Documents\\Project Anime\\static'
@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        anime_type = request.form['anime_type']
        anime_name = request.form['anime_name']
        num_recommendations = int(request.form['num_recommendations'])
        recommendations = get_anime_recommendations(anime_type , anime_name, num_recommendations)
        return render_template('results.html', recommendations=recommendations)
    return render_template('index.html' ,anime_names = anilist)

if __name__ == '__main__':
    app.run(debug=True)

