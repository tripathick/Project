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
sns.jointplot(x='rating' , y='num of ratings' , data=ratings , alpha=0.5)
sns.show()