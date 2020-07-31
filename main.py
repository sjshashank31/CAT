import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


df = pd.read_csv('2.csv')
y=[]

for i in df:
    y.append(sum(df[i]))
print(y)
mean = np.mean(y)
min = np.min(y)
max = np.max(y)
print(mean)
print(min)
print(max)
l=[]
for i in y:
    l.append((mean-i)/(max-min))
    

print(len(l))

sq = 10**len(str(df.shape[1]))
print(sq)

x = [i/sq for i in range(df.shape[1])]

k= 3

df1 = pd.DataFrame({
            'x': x,
            'y': l
}
)

kmeans = KMeans(n_clusters=3)
kmeans.fit(df1)
new_dict = []
labels = kmeans.predict(df1)
for j in df1['y']:
    if j>-0.2 and j<0.2:
        new_dict.append("Medium")
    elif j>0.2:
        new_dict.append("Hard")
    else:
        new_dict.append("Easy")


level = pd.DataFrame({'level': new_dict})
level.to_csv('level.csv')

colmap = {1:'r', 2:'b', 3:'g'}
fig = plt.figure(figsize=(5,10))
colors = map(lambda a: colmap[a+1], labels)
colors1 = list(colors)
plt.scatter(df1['x'], df1['y'], color=colors1, edgecolor='k')
