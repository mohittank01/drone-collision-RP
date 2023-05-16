import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
import pyproj
import math
import matplotlib.pyplot as plt
from cartes.crs import OSGB
import cartopy.crs as ccrs
from traffic.data import airports
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

airport = "LHR"
distance_from_airport = "1.0"

drone_data = pd.read_csv(airport + "/Drone_Collisions_" + distance_from_airport + "_km/Depart/Batch_Testing/Average_Collisions.csv")
drone_data = drone_data.apply(pd.to_numeric, errors='coerce')

total_MC_runs = int(drone_data['longitude'].iloc[-3])


sim_index = [None] * total_MC_runs

idx = 0

# Identifying rows which have text in them with LOCAL COLLISIONS
for i in range(1,len(drone_data['run_number'])+1):
    if drone_data['run_number'].iloc[i] != drone_data['run_number'].iloc[i-1]:
        if math.isnan(drone_data['run_number'].iloc[i-1]):
            break
        sim_index[idx] = i-1
        idx += 1

# Dropping data which has the text in them
drone_data = drone_data.drop(drone_data.index[sim_index])
drone_data = drone_data.drop(drone_data.index[[-1,-2,-3,-4,-5]])

p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
#drone_long, drone_lat = p(drone_data['longitude'],drone_data['latitude'],inverse=True)

drone_long = list(drone_data['longitude'])
drone_lat = list(drone_data['latitude'])
drone_altitude = list(drone_data['altitude'])

# Aligning lonitude,latitude and altitude into one array
points = np.empty([len(drone_long),3])
for i in range(0,len(drone_long)):
    points[i, 0] = drone_long[i]
    points[i, 1] = drone_lat[i]
    points[i, 2] = drone_altitude[i]

db = DBSCAN(eps=30, min_samples=10).fit(points)

print(len(Counter(db.labels_))-1)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], c=db.labels_)
plt.show()
