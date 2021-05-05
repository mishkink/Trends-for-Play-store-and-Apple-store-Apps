import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import sys
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

## %!!!!!!!!!! Data Reading !!!!!!!!!!!%
data = pd.read_csv('/Users/mishkinkhunger/Desktop/Capstone/Dataset/googleplaystore.csv')
reviews = pd.read_csv('/Users/mishkinkhunger/Desktop/Capstone/Dataset/googleplaystore_user_reviews.csv')
print(data.columns)
# print(reviews.columns)
#
# print(data.head())
# print(reviews.head())

data = data.reset_index(drop=True)
reviews = reviews.reset_index(drop=True)

df = pd.concat([data, reviews], axis=1, join="inner")

df = pd.read_csv('/Users/mishkinkhunger/Desktop/Capstone/Dataset/googleplaystore.csv')
print(df.loc[df['App'].values == 'Sketch - Draw & Paint'])
print(df.columns)
print(df.info)
## %!!!!!!!!!! Data Cleaning !!!!!!!!!!!%
print('Price original')
print(df.Price.unique())

print('Price after processing:')
df.Price = df.Price.apply(lambda x: str(x).replace("$",""))
print(df.Price.unique())

print('Basic Statistics')
df.Reviews = pd.to_numeric(df.Reviews, errors='coerce')
df.Price = pd.to_numeric(df.Price, errors='coerce')
df.Rating = pd.to_numeric(df.Rating, errors='coerce')
print(df.dtypes)

## %!!!!!!!!!! Data Mining !!!!!!!!!!!%

print(df.describe().T)
print('Correlation between features')
# print(df.corr())

plt.figure(figsize=(10,11))
sns.heatmap(df.corr(),annot=True, cmap="coolwarm")
plt.plot()
plt.show()

# Missing values
print('No. of Missing values:')
print(df.isna().sum())

print('Value counts of ratings')
print(df["Rating"].value_counts())

df = df[df["Rating"] != 19.0]

print('Value counts of ratings')
print(df["Rating"].value_counts())

## %!!!!!!!!!! Data Preprocessing !!!!!!!!!!!%

#Treating Null values by mean
df.isna().sum()
print(df.describe().T)
df["Rating"] = df["Rating"].fillna(4.193338)
print(df.isna().sum())

# Grouping according to Category
df.groupby(["Category"]).std()["Price"]

## %!!!!!!!!!! Data Visualization !!!!!!!!!!!%

# Distribution of Rating variable
plt.figure(figsize=(25,5))
fig = sns.distplot(df["Rating"], bins = "auto", color = "black")
fig.set_xticklabels(fig.get_xticklabels(),rotation = 90)

plt.show()

# Grouping according to category
plt.figure(figsize=(20,5))
fig = df.groupby(["Category"]).std()["Price"].plot(kind = "bar")
fig.set_xticklabels(fig.get_xticklabels(),rotation = 90)
plt.show()

# Average ratings in each category


df.groupby(["Category"])["Rating"].mean().plot(kind = "bar", title = "Mean of Categories", color = "red")
plt.xlabel("Category");
plt.ylabel("Count");
plt.show()

# Most installed applications
df.groupby(["Category"])["Installs"].count().sort_values(ascending = False)
df["Category"].value_counts()
#percentage wise installion
(df["Category"].value_counts() / len(df["Category"])) * 100

fig1,ax1 = plt.subplots(figsize=(25,25))

mylabels = df["Category"].value_counts().index

ax1.pie(df["Category"].value_counts(), labels = mylabels,autopct='%1.1f%%');
plt.show()


df.groupby(["Category"])["Installs"].count().sort_values(ascending = False).head(20).plot(kind = "bar", title = "Top 20 Most Uploaded Categories", color='red')
plt.xlabel("Category");
plt.ylabel("Count");
plt.show()

new = df['Price']
x = (new == 0).sum()
y = len(new) - x

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# langs = ['Free Apps','Paid Apps']
# x_pos = np.arange(len(langs))
# nums = [x,y]
# ax.bar(langs,nums)
# plt.title('My title')
# plt.xlabel('categories')
# plt.ylabel('values')
# plt.xticks(x_pos, langs)
# plt.show()

# creating the dataset
data = {'Free Apps': x, 'Paid Apps': y}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color='green',
        width=0.4)

plt.xlabel("Type of Apps")
plt.ylabel("Number of Records")
plt.title("Number of Installation of Paid and Free Apps")
plt.show()








