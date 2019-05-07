#!/usr/bin/python3

#Handle Parameters sending from php
import sys

titleList = []
i=1
while i<len(sys.argv):
    titleList.append(sys.argv[i])
    i += 1

title = " ".join(titleList)
#print (title)

#Access to mySQL search database
import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="username",
  passwd="password",
  database="table"
)

#Convert SQL to Pandas Data structure
import pandas as pd
import numpy as np

df = pd.read_sql('SELECT Title, Overview, Popularity FROM movie2 WHERE Overview <> "" and Title <> ""', con=mydb)
#print (df.shape)
#print (df['Title'].head())

#Build Tf-idf matrix for overview of each movie by TfIdfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Overview'])
#print (tfidf_matrix.shape)

#Compute the cosine similarity matrix
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

#Function that takes in movie title as input and outputs ten most similar movies in terms of tf-idf
def get_tfidf(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    get_tfidf.scores = sim_scores
    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices]

# Function that takes in movie title as input and outputs popularity of ten most similar movies
def get_pop(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['Popularity'].iloc[movie_indices]

movieSimiliar = get_tfidf(title)

#List for weighted similiar score
weight_sim=[]
for x in get_tfidf.scores:
    weight_scores = x[1]*0.8
    weight_sim.append(weight_scores)
#print (weight_sim)

#List for weighted popularity score
weight_pop=[]
for x in get_pop(title):
    weight_scores = x*0.01*0.2
    weight_pop.append(weight_scores)
#print (weight_pop)

#Add the two number up as the final weighted score interms of similarity and popularity values
weighted_scores = np.add(weight_sim, weight_pop)
#print (weighted_scores)

#List for index of ten most similar movies
weighted_index=[]
for x in get_tfidf.scores:
    index = x[0]
    weighted_index.append(index)
#print (weighted_index)

#List for index and weighted score pairwise
weighted_scoresIndex = list(zip(weighted_index, weighted_scores))

#Sort movies according to weighted scores in decrease order
weighted_scoresIndex = sorted(weighted_scoresIndex, key = lambda x: x[1], reverse=True)
#print (weighted_scoresIndex)


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = weighted_scoresIndex
    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices]

movieRec = get_recommendations(title)

movie1 = movieRec.iloc[0]
movie2 = movieRec.iloc[1]
movie3 = movieRec.iloc[2]
movie4 = movieRec.iloc[3]
movie5 = movieRec.iloc[4]
movie6 = movieRec.iloc[5]
movie7 = movieRec.iloc[6]
movie8 = movieRec.iloc[7]
movie9 = movieRec.iloc[8]
movie10 = movieRec.iloc[9]
print ('You may also like: 1. '+ movie1 + ' 2. ' + movie2 + ' 3. ' + movie3 + ' 4. ' + movie4 + ' 5. ' + movie5 + ' 6. ' + movie6 + ' 7. ' + movie7 + ' 8. ' + movie8 + ' 9. ' + movie9 + ' 10. ' + movie10)
