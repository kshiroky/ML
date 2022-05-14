from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

#вводим данные
data = pd.read_csv('/home/nikolai/Downloads/ML_9.csv')
x = pd.Series(data['X'])
y = pd.Series(data['Y'])
new_data = pd.merge(x,y, right_index=True, left_index=True)

#обучаем классификатор
result = KMeans(n_clusters = 3, init = np.array([[11.8, 11.6], [8.5, 9.83], [14.0, 14.5]]), max_iter = 100, n_init = 1).fit(new_data)

#находим распределение по кластерам и записываем его в таблицу с данными
a = result.labels_
cluster = pd.Series(a, name='cluster')
res_data = new_data.merge(cluster, right_index = True, left_index = True)

#находим расстояние до нулевого центроида
dist = result.transform(new_data)

print(type(dist))
#далее индексы - порядковые номера записей в таблице, которых классификатор отнес к кластеру 0 (начиная с нуля)
indexes = [2,3,7,10,13]
distance = []
for each in indexes:
    ds = np.array(dist[each])
    distance.append(ds)
print(distance)

#из рещультата предыдущего запроса нужно взять первые числа в каждом из array и найти их среднее
res_dist = (1.26491106 + 7.65506368 + 1.84390889 + 1.78885438 + 6.26099034)/5
print(res_dist)