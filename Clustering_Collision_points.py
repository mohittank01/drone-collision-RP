import numpy as np
from sklearn.cluster import DBSCAN
from haversine import inverse_haversine, Direction, haversine
import pandas as pd
import pyproj
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from cartes.crs import OSGB
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import itertools


def boundary(coords, distance):
    west = inverse_haversine(coords, distance, Direction.WEST)
    east = inverse_haversine(coords, distance, Direction.EAST)
    north = inverse_haversine(coords, distance, Direction.NORTH)
    south = inverse_haversine(coords, distance, Direction.SOUTH)
    return [west[1], south[0], east[1], north[0]]

#returns average_collision,average_percentage_collision,average_time,drone_data
def data_access(airport,distance_from_airport,drone_model,depart_arrive):
    drone_data = pd.read_csv(
        airport + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model + "/" + depart_arrive + "/Batch_Testing/Average_Collisions.csv")
    drone_data = drone_data.apply(pd.to_numeric, errors='coerce')
    total_MC_runs = int(drone_data['longitude'].iloc[-3])
    total_collisions = drone_data['longitude'].iloc[-5]
    total_simulations = drone_data['longitude'].iloc[-4]
    average_collision = drone_data['longitude'].iloc[-2]
    average_percent_collisions = drone_data['longitude'].iloc[-1]

    sim_index = [None] * total_MC_runs

    idx = 0

    # Identifying rows which have text in them with LOCAL COLLISIONS
    for i in range(1, len(drone_data['run_number']) + 1):
        if drone_data['run_number'].iloc[i] != drone_data['run_number'].iloc[i - 1]:
            if math.isnan(drone_data['run_number'].iloc[i - 1]):
                break
            sim_index[idx] = i - 1
            idx += 1

    # Dropping data which has the text in them
    drone_data.drop(drone_data.index[sim_index], inplace=True)
    drone_data.drop(drone_data.index[[-1, -2, -3, -4, -5]], inplace=True)
    drone_data.reset_index(drop=True, inplace=True)

    sim_index = [None] * (total_MC_runs - 1)

    idx = 0

    # Identifying rows which have text in them with LOCAL COLLISIONS
    for i in range(1, len(drone_data['run_number'])):
        if drone_data['run_number'].iloc[i] != drone_data['run_number'].iloc[i - 1]:
            if math.isnan(drone_data['run_number'].iloc[i - 1]):
                break
            sim_index[idx] = i
            idx += 1

    average_time = sum(drone_data['time'])/len(drone_data['time'])
    return average_collision,average_percent_collisions,average_time,drone_data,sim_index

airport = "LHR"
distance_from_airport = "1.0"
drone_model = 'Mavic_3' # Either 'Mavic_3' or 'Mini_2'
depart_arrive = "Depart" # Either "Depart" or "Arrival"

heathrow = [51.471305, -0.460861]
gatwick = [51.153871, -0.187176]

bounds_LHR = boundary(heathrow, 5)
bounds_LGW = boundary(gatwick, 5)

average_collision,average_percent_collision,average_time,drone_data,sim_index = data_access(airport,distance_from_airport,drone_model,depart_arrive)

drone_long = list(drone_data['longitude'])
drone_lat = list(drone_data['latitude'])
drone_altitude = list(drone_data['altitude'])

# Aligning lonitude,latitude and altitude into one array
points = np.empty([len(drone_long),3])
for i in range(0,len(drone_long)):
    points[i, 0] = drone_long[i]
    points[i, 1] = drone_lat[i]
    points[i, 2] = drone_altitude[i]

db = DBSCAN(eps=50, min_samples=15).fit(points)

print(len(Counter(db.labels_))-1)

p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
drone_long_deg, drone_lat_deg = p(drone_long,drone_lat,inverse=True)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(drone_long_deg, drone_lat_deg, points[:,2], c=db.labels_)
plt.title("Clusters of Collisions between Aircraft and Drone - " + airport)
ax.set_xlabel("Longitude / $\circ$")
ax.set_ylabel('Latitude / $\circ$')
ax.set_zlabel('Altitude / m')
plt.show()

fig = plt.figure()
plot_label = 1
plt.plot(drone_data['drone_speed'].iloc[0:sim_index[0]],drone_data['aircraft_speed'].iloc[0:sim_index[0]],'x', label=plot_label)
for i in range(0,len(sim_index)-1):
    plot_label += 1
    plt.plot(drone_data['drone_speed'].iloc[sim_index[i]:sim_index[i+1]],drone_data['aircraft_speed'].iloc[sim_index[i]:sim_index[i+1]],'x',label=plot_label)
plot_label += 1
plt.plot(drone_data['drone_speed'].iloc[sim_index[-1]:len(drone_data['drone_speed'])],drone_data['aircraft_speed'].iloc[sim_index[-1]:len(drone_data['drone_speed'])],'x',label=plot_label)
plt.title("Aircraft Groundspeed vs Drone Groundspeed")
plt.xlabel("Drone Groundspeed / m/s")
plt.ylabel("Aircraft Groundspeed / m/s")
plt.legend(title='Run No.')
plt.show()



########################################################################################################

############################################# TESTING ##################################################
"""
fig = plt.figure()
#ax = Axes3D(fig, xlim=[bounds_LHR[0], bounds_LHR[2]], ylim=[bounds_LHR[1], bounds_LHR[3]])
ax = Axes3D(fig, xlim=[-180, 180], ylim=[-90, 90])
ax.set_zlim(bottom=0)
#ax.set_xlim3d(left=bounds_LHR[0], right=bounds_LHR[2])
#ax.set_ylim3d(bottom=bounds_LHR[1], top=bounds_LHR[3])

#ax = fig.add_subplot(111, projection='3d')



concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

target_projection = ccrs.PlateCarree()

feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m')
geoms = feature.geometries()

geoms = [target_projection.project_geometry(geom, feature.crs)
         for geom in geoms]

paths = concat(geos_to_path(geom) for geom in geoms)

polys = concat(path.to_polygons() for path in paths)

lc = PolyCollection(polys, edgecolor='black',
                    facecolor='green', closed=False)

ax.scatter(drone_long_deg, drone_lat_deg, points[:,2], c=db.labels_)
ax.add_collection3d(lc)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')
#ax.set_xlim([bounds_LHR[0], bounds_LHR[2]])
#ax.set_ylim([bounds_LHR[1], bounds_LHR[3]])
ax.set_zlim(bottom=0)

plt.show()
"""


