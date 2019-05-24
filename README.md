# Movie Recommender System
A Movie Recommender Systems Based on Tf-idf and Popularity.

## Overview
We developed this content-based movie recommender based on two attributes, overview and popularity. Firstly, we calculate similarities between any two movies by their overview tf-idf vectors. Script *rec.py* stops here. This is a basic recommender only evaluated by overview. Factors, like rating, actors, publish year and popularity, etc. can also contribute to the recmmended result. Script *rec_pop.py* add one more feature, popularity, as input to adjust final results. The weighted similarity score consist of 80% tf-idf vector and 20% popularity.

## Scripts in detail
- rec.html  
The script is a basic user interface for input.
```html
<h1>Please Enter Movie Title to Get Recommendation</h1>
<form class="search" action="rec.php" method="get" style="margin:auto;max-width:300px">
  <input type="text" placeholder="Search Your Favourite Movie" name="user_query">
  <button type="submit" name="search"><i class="fa fa-search"></i></button>
</form>
```
- rec.php  
The php script gets user input, forwards the user input variable to the python recommender script and calls the recommender script to run. Once the the recommender script finishing running, it output the final result to user by browser.
```php
<?php 
//Form data handling
$input = $_GET["user_query"];

//Execute python script
$command = escapeshellcmd('./rec_pop.py ' . $input);
$output = shell_exec($command);
?>
<h2> <?php echo $output ?> </h2>
```

- rec.py  
This is the main script for recommender system. We connected MySQL databse, read dataset into pandas dataframe, build a tf-idf matrix for overview, compute the cosine similarity matrix and get recommendations by the cosine similarity matrix.
```python
import sys
titleList = []
i=1
while i<len(sys.argv):
	titleList.append(sys.argv[i])
	i += 1
title = " ".join(titleList)
```
```python
import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="username",
  passwd="password",
  database="table"
)
import pandas as pd
import numpy as np
df = pd.read_sql('SELECT Title, Overview, Popularity FROM movie2 WHERE Overview <> "" and Title <> ""', con=mydb)
```
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Overview'])
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
```
```python
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices]
```
```python
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
```
- rec_pop.py  
The *rec_pop.py* is independent from *rec.py*. Everything keeps the same until the *get_recommendations* function. We replaced *get_recommendations* to *get_tfidf* since we will not get results only by overview vector.
```python
def get_tfidf(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    get_tfidf.scores = sim_scores
    movie_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[movie_indices]
def get_pop(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['Popularity'].iloc[movie_indices]
movieSimiliar = get_tfidf(title)
```
```python
weight_sim=[]
for x in get_tfidf.scores:
    weight_scores = x[1]*0.8
    weight_sim.append(weight_scores)
weight_pop=[]
for x in get_pop(title):
    weight_scores = x*0.01*0.2
    weight_pop.append(weight_scores)
weighted_scores = np.add(weight_sim, weight_pop)
weighted_index=[]
for x in get_tfidf.scores:
    index = x[0]
    weighted_index.append(index)
weighted_scoresIndex = list(zip(weighted_index, weighted_scores))
weighted_scoresIndex = sorted(weighted_scoresIndex, key = lambda x: x[1], reverse=True)
```
```python
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
```
## Methodology on blog
https://cwang.netlify.com/post/movie_recommendation_engine/
