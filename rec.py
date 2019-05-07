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

#Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
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
