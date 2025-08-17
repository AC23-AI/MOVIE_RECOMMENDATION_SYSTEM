import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk. stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
#--------------------------------------------------------------------------------------------------------------
#_______________________________________________LOADING OF DATA SET____________________________________________
#---------------------------------------------------------------------------------------------------------------
movies = pd.read_csv("C:/Users/USER/Desktop/MRS/tmdb_5000_movies.csv")
credits = pd.read_csv("C:/Users/USER/Desktop/MRS/tmdb_5000_credits.csv")
movies = movies.merge(credits,on='title')

#---------------------------------------------------------------------------------------------------------------
#_______________________________________________PREPROCESSING ON DATA SET_______________________________________
#---------------------------------------------------------------------------------------------------------------
# Useful columns = genres, keyword, title, overview , cast , crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()

movies.isnull().sum()
movies.dropna(inplace=True)

#Function to extract essential keywords from various dictionaries
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Function to extract  top 3 cast from the dataset
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

# Fetch the data  of director from the dataset
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Performing Stemming
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i)) 
    return " ".join(y)

# recommend Top 5 movies
def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]
    distances =  similarity[movie_index]
    movies_list  = sorted(list(enumerate(distances)),reverse= True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_movies.iloc[i[0]].title)

    
        

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(convert3)

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert string into  list
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# Remove extra space in between the list elements
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# Formation of new column  'tags' 
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x:" ".join(x))
movies['tags'] = movies['tags'].apply(lambda x :x.lower())

new_movies = movies[['movie_id','title','tags']]
new_movies['tags'] = new_movies['tags'].apply(stem)

#--------------------------------------------------------------------------------------------------------------------
#_______________________________________________VECTORIZATION____________________________________________________
#-----------------------------------------------------------------------------------------------------------------
cv = CountVectorizer(max_features=5000,stop_words = 'english')
vectors = cv.fit_transform(new_movies["tags"]).toarray()
cv.get_feature_names_out()

similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse= True,key =lambda x:x[1])[1:6]

recommend('Resident Evil')
#----------------------------------------------------------------------------------------------------------------
#______________________________________________SAVING MODEL INTO PICKLE FILE__________________________________
#-----------------------------------------------------------------------------------------------------------------

pickle.dump(new_movies,open("movies.pkl","wb"))
pickle.dump(new_movies.to_dict(),open("movies_dict.pkl","wb"))
pickle.dump(similarity,open("similarity.pkl","wb"))











