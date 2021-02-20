#Creating Movie Recomendation system
import numpy as np 
import pandas as pd 
import warnings

warnings.filterwarnings('ignore')

columns_names = ["user_id" , "item_id" , "rating" , "timestamp"]


df = pd.read_csv('u.data',sep='\t' , names=columns_names)

#print(df.head())

#print(df.shape)

print(df['user_id'].nunique())
print(df['item_id'].nunique())

movies_titles=pd.read_csv('u.item',sep='\|',encoding = "ISO-8859-1",header=None)
movies_titles = movies_titles[[0,1]]
movies_titles.columns = ['item_id' , 'title']
#print(movies_titles.head())


#mrege two dataframe
df = pd.merge(df,movies_titles, on="item_id")
#print(df)
#print(df.tail())


#part:3
#Exploratory data Analysis:

import matplotlib.pyplot as plt 
import seaborn as sns  # for data visualisation
sns.set_style('white')

#print(df.groupby('title').mean()['rating'].sort_values(ascending=False).head())
#print(df.groupby('title').count()['rating'].sort_values(ascending=False))

#Creating data fframe of rating...
ratings = pd.DataFrame(df.groupby('title').mean(['rating']))
#print(rating.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
#print(ratings.sort_values(by='rating' , ascending=False ))

plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'], bins = 70)
#plt.show()

# Histogram of ratings
plt.hist(ratings['rating'],bins = 70)
#plt.show()

# Graph of rating Vs actual rating of movie
#sns.jointplot(x='rating' , y='num of ratings' , data=ratings , alpha=0.5)
#sns.show()
########## MOVIE RECOMENDATION SYSTEM  #####################

#print(df.head())
moviemat = df.pivot_table(index = "user_id" , columns = "title" , values = "rating")
#print(moviemat)

# which movie has highest ratinng 
#print(ratings.sort_values('num of ratings' , ascending=False).head()) 

# Let's checking userwise rating
starwars_user_ratings = moviemat['Star Wars (1977)']
#print(starwars_user_ratings.head())


# compare movie rating of Star Wars from other  movie title..
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
#print(similar_to_starwars)

       #convert this seriies of comparision into dataframe..
corr_starwars = pd.DataFrame(similar_to_starwars , columns = ['correlation'])
#print(corr_starwars.head())
#print(corr_starwars.dropna(inplace=True))

#print(corr_starwars.sort_values('correlation',ascending=False).head(10))

###### Filteration of high Rated Movie  ##########
corr_starwars = corr_starwars.join(ratings['num of ratings'])
#print(corr_starwars.head())

#print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False))

######## PREDICT FUNCTIION #################
# For take user input as movie name , and give the output as recomendation of movie

def predict_movies(movie_name):
	movie_user_ratings = moviemat[movie_name]
	similar_to_movie = moviemat.corrwith(movie_user_ratings)

	corr_movie = pd.DataFrame(similar_to_movie, columns=['correlation'])
	corr_movie.dropna(inplace = True)

	corr_movie = corr_movie.join(ratings['num of ratings'])
	predictions = corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending = False)

	return predictions
    
predictions = predict_movies('Titanic (1997)')
print(predictions.head())
