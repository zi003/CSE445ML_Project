##Implementing DBSCAN on dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

data = pd.read_csv("cleaned_data.csv")

##dropping redundant columns which specifying each screen time
data = data.drop(columns = ['user_id','phone_usage_hours','laptop_usage_hours','tablet_usage_hours','social_media_hours'])

#shuffling the data 
data = data.sample(frac = 1, random_state = 10)

#defining X(input features) and Y(output target)
X = data.drop(columns = 'mental_health_score')
Y = data['mental_health_score']

##feature scaling 
scaler = StandardScaler().set_output(transform='pandas')
data_scaled = scaler.fit_transform(X)

#defining the parameters 
number_of_features = 22
min_pts = 2 * number_of_features
neigh = NearestNeighbors(n_neighbors = min_pts)  #initializing model with 'min_pts' number of neighbours
neigh.fit(data_scaled) ##fitting model with my dataset 
distance, _ = neigh.kneighbors(data_scaled)   ##finding k nearest neighbours for each sample ignoring the indices of the nearest neighbors
distance = np.sort(distance[:, min_pts-1])




#from graph elbow point can be taken as average of 29 and 34.4 (y value)
kneedle = KneeLocator(range(len(distance)), distance, curve='convex', direction='increasing')
epsilon = distance[kneedle.knee]
dbscan = DBSCAN(eps = epsilon,min_samples = min_pts)
clusters = dbscan.fit_predict(data_scaled)
clustered_data = data_scaled.copy()
clustered_data['cluster'] = clusters
clustered_data['mental_health_score'] = Y
print(np.unique(clusters))