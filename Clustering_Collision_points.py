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
from scipy.interpolate import UnivariateSpline
from traffic.data import airports
from scipy import interpolate

plt.close('all')


def boundary(coords, distance):
    west = inverse_haversine(coords, distance, Direction.WEST)
    east = inverse_haversine(coords, distance, Direction.EAST)
    north = inverse_haversine(coords, distance, Direction.NORTH)
    south = inverse_haversine(coords, distance, Direction.SOUTH)
    return [west[1], south[0], east[1], north[0]]

#returns average_collision,average_percent_collisions,average_time,drone_data,drone_pos,sim_index,average_altitude
def data_access(airport,distance_from_airport,drone_model,depart_arrive):
    drone_data = pd.read_csv(
        airport + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model + "/" + depart_arrive + "/Batch_Testing/Average_Collisions.csv")
    drone_data = drone_data.apply(pd.to_numeric, errors='coerce')
    total_MC_runs = int(drone_data['longitude'].iloc[-3])
    total_collisions = drone_data['longitude'].iloc[-5]
    total_simulations = drone_data['longitude'].iloc[-4]
    average_collision = drone_data['longitude'].iloc[-2]
    average_percent_collisions = drone_data['longitude'].iloc[-1]

    drone_pos = pd.read_csv(airport + "/drone_inital_positions_" + distance_from_airport + "km.csv")
    drone_pos.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
    drone_pos_long = list(drone_pos['longitude'])
    drone_pos_lat = list(drone_pos['latitude'])
    drone_pos['longitude'], drone_pos['latitude'] = p(drone_pos_long, drone_pos_lat, inverse=True)

    # Conversion from m to feet
    drone_data['altitude'] = drone_data['altitude'] * 3.28084

    drone_long = list(drone_data['longitude'])
    drone_lat = list(drone_data['latitude'])
    drone_data['longitude'], drone_data['latitude'] = p(drone_long, drone_lat, inverse=True)


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

    if average_collision == 0:
        average_time = float('nan')
        average_altitude = float('nan')
    else:
        average_time = sum(drone_data['time'])/len(drone_data['time'])
        average_altitude = sum(drone_data['altitude'])/len(drone_data['altitude'])

    return average_collision,average_percent_collisions,average_time,drone_data,drone_pos,sim_index,average_altitude

def distance_comparison(airport,drone_model,depart_arrive):
    distance_from_airport = np.arange(1.0, 5.5, 0.5).tolist()

    heathrow = [51.471305, -0.460861]
    gatwick = [51.153871, -0.187176]

    bounds_LHR = boundary(heathrow, 5)
    bounds_LGW = boundary(gatwick, 5)

    # Plotting where drones come from
    fig, ax = plt.subplots(
        1,figsize=(10, 10),
        subplot_kw=dict(projection=OSGB())
    )
    if airport == "LHR":
        airport_code = "EGLL"
        bounds = bounds_LHR

    if airport == "LGW":
        airport_code = "EGKK"
        bounds = bounds_LGW

    airports[airport_code].plot(ax)

    for i in range(0, len(distance_from_airport)):
        distance = str(distance_from_airport[i])
        _, _, _, drone_data, drone_pos, _, _ = data_access(airport,distance,drone_model,depart_arrive)

        drone_loc = list(drone_data['drone_index'].astype(int))
        counter = Counter(drone_loc)
        drone_indexes = list(counter.keys())
        index_values = list(counter.values())
        new_list = list(zip(drone_indexes, index_values))
        new_list.sort()
        drone_indexes, index_values = list(zip(*new_list))
        drone_pos_total = pd.DataFrame({"index": drone_indexes, "total": index_values}).merge(drone_pos, on='index', how='left')

        plt.scatter(drone_pos_total['longitude'], drone_pos_total['latitude'], s=drone_pos_total['total'], alpha=0.5,
                    transform=ccrs.PlateCarree(), label=distance)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Areas of where Colliding Drones come from\n' + airport + " - " + depart_arrive)
    ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]))
    legend = plt.legend(title='Distance from Airport Center - km',fontsize=15, loc='lower center',ncol=3)
    legend.get_title().set_fontsize('16')
    ax.title.set_fontsize(20)
    plt.show()

    # Plotting Groundspeed comparision
    plt.figure()
    for i in range(0, len(distance_from_airport)):
        distance = str(distance_from_airport[i])
        _, _, _, drone_data, _, sim_index,_ = data_access(airport,distance,drone_model,depart_arrive)

        # Plotting Airspeed comparison
        plt.plot(drone_data['drone_speed'].iloc[0:sim_index[0]], drone_data['aircraft_speed'].iloc[0:sim_index[0]], 'x',
                 label=distance)

    plt.title("Aircraft Groundspeed vs Drone Groundspeed\n" + airport + " - " + depart_arrive)
    plt.xlabel("Drone Groundspeed / m/s")
    plt.ylabel("Aircraft Groundspeed / m/s")
    plt.legend(title='Distance from\nAirport Center - km',bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def depart_arrival_comparison(airport,drone_model):
    distance_from_airport = np.arange(1.0, 5.5, 0.5).tolist()

    all_avg_col_percent_depart = [None] * len(distance_from_airport)
    all_avg_time_depart = [None] * len(distance_from_airport)
    all_avg_altitude_depart = [None] * len(distance_from_airport)

    all_avg_col_percent_arrive = [None] * len(distance_from_airport)
    all_avg_time_arrive = [None] * len(distance_from_airport)
    all_avg_altitude_arrive = [None] * len(distance_from_airport)

    for i in range(0, len(distance_from_airport)):
        distance = str(distance_from_airport[i])
        _, average_percent_collision_d, average_time_d, _, _, _, average_altitude_d = data_access(airport, distance,
                                                                                                  drone_model, "Depart")
        _, average_percent_collision_a, average_time_a, _, _, _, average_altitude_a = data_access(airport, distance,
                                                                                                  drone_model,
                                                                                                  "Arrival")

        all_avg_col_percent_depart[i] = average_percent_collision_d
        all_avg_time_depart[i] = average_time_d
        all_avg_altitude_depart[i] = average_altitude_d

        all_avg_col_percent_arrive[i] = average_percent_collision_a
        all_avg_time_arrive[i] = average_time_a
        all_avg_altitude_arrive[i] = average_altitude_a

    discretised_distance = np.linspace(min(distance_from_airport), max(distance_from_airport), 1000)
    # Plotting Average Collision percent against distance

    # spline = UnivariateSpline(distance_from_airport,all_avg_col_percent)
    avg_col_percent_spline_d = interpolate.make_interp_spline(distance_from_airport, all_avg_col_percent_depart)
    avg_col_percent_spline_a = interpolate.make_interp_spline(distance_from_airport, all_avg_col_percent_arrive)
    plt.figure()
    plt.plot(discretised_distance, avg_col_percent_spline_d(discretised_distance), '-b', label="Depart")
    plt.plot(distance_from_airport, all_avg_col_percent_depart, 'bo')
    plt.plot(discretised_distance, avg_col_percent_spline_a(discretised_distance), '-r', label="Arrival")
    plt.plot(distance_from_airport, all_avg_col_percent_arrive, 'ro')
    plt.title("Average Collision Percentage against \n Drone distance from Airport Center - " + airport + " - " + drone_model)
    plt.xlabel("Drone distance from center of airport / km")
    plt.ylabel("Average collision percentage / %")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # THIS IS TO ELIMINATE TIMES WHERE THERE WERE NO COLLISIONS AT ALL IN THE SIMULATIONS
    if np.isnan(all_avg_time_arrive).any() or np.isnan(all_avg_time_depart).any():
        valid_indices_a = np.logical_not(np.isnan(all_avg_time_arrive))
        valid_indices_d = np.logical_not(np.isnan(all_avg_time_depart))

        distance_from_airport_a = np.array(distance_from_airport)[valid_indices_a]
        distance_from_airport_d = np.array(distance_from_airport)[valid_indices_d]

        all_avg_time_arrive = np.array(all_avg_time_arrive)[valid_indices_a]
        all_avg_time_depart = np.array(all_avg_time_depart)[valid_indices_d]

        all_avg_altitude_arrive = np.array(all_avg_altitude_arrive)[valid_indices_a]
        all_avg_altitude_depart = np.array(all_avg_altitude_depart)[valid_indices_d]

        discretised_distance_a = np.linspace(min(distance_from_airport_a), max(distance_from_airport_a), 1000)
        discretised_distance_d = np.linspace(min(distance_from_airport_d), max(distance_from_airport_d), 1000)

    else:
        discretised_distance_a = np.linspace(min(distance_from_airport), max(distance_from_airport), 1000)
        discretised_distance_d = np.linspace(min(distance_from_airport), max(distance_from_airport), 1000)
        distance_from_airport_d = distance_from_airport
        distance_from_airport_a = distance_from_airport

    # Plotting change in time of collision
    avg_time_fit_d = np.poly1d(np.polyfit(distance_from_airport_d, all_avg_time_depart, 3))
    avg_time_fit_a = np.poly1d(np.polyfit(distance_from_airport_a, all_avg_time_arrive, 3))
    plt.figure()
    plt.plot(discretised_distance_d, avg_time_fit_d(discretised_distance_d), '-b', label="Depart")
    plt.plot(distance_from_airport_d, all_avg_time_depart, 'bo')
    plt.plot(discretised_distance_a, avg_time_fit_a(discretised_distance_a), '-r', label="Arrival")
    plt.plot(distance_from_airport_a, all_avg_time_arrive, 'ro')
    plt.title("Average Collision Time against \n Drone distance from Airport Center - " + airport + " - " + drone_model)
    plt.xlabel("Drone distance from center of airport / km")
    plt.ylabel("Average collision time / s")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting average altitude of collisions against distance
    avg_altitude_fit_d = np.poly1d(np.polyfit(distance_from_airport_d, all_avg_altitude_depart, 3))
    avg_altitude_fit_a = np.poly1d(np.polyfit(distance_from_airport_a, all_avg_altitude_arrive, 3))
    # avg_altitude_fit_d = UnivariateSpline(distance_from_airport_d,all_avg_altitude_depart)
    # avg_altitude_fit_a = UnivariateSpline(distance_from_airport_a, all_avg_altitude_arrive)
    plt.figure()
    plt.plot(discretised_distance_d, avg_altitude_fit_d(discretised_distance_d), '-b', label="Depart")
    plt.plot(distance_from_airport_d, all_avg_altitude_depart, 'bo')
    plt.plot(discretised_distance_a, avg_altitude_fit_a(discretised_distance_a), '-r', label="Arrival")
    plt.plot(distance_from_airport_a, all_avg_altitude_arrive, 'ro')
    plt.title("Average Collision Altitude against \n Drone distance from Airport Center - " + airport + " - " + drone_model)
    plt.xlabel("Drone distance from center of airport / km")
    plt.ylabel("Average collision altitude / ft")
    plt.legend()
    plt.tight_layout()
    plt.show()

def drone_comparison(airport):
    distance_from_airport = np.arange(1.0, 5.5, 0.5).tolist()

    all_avg_col_percent_depart = [None] * len(distance_from_airport)
    all_avg_time_depart = [None] * len(distance_from_airport)
    all_avg_altitude_depart = [None] * len(distance_from_airport)

    all_avg_col_percent_arrive = [None] * len(distance_from_airport)
    all_avg_time_arrive = [None] * len(distance_from_airport)
    all_avg_altitude_arrive = [None] * len(distance_from_airport)

    for i in range(0, len(distance_from_airport)):
        distance = str(distance_from_airport[i])
        _, average_percent_collision_d, average_time_d, _, _, _, average_altitude_d = data_access(airport, distance,
                                                                                                  drone_model, "Depart")
        _, average_percent_collision_a, average_time_a, _, _, _, average_altitude_a = data_access(airport, distance,
                                                                                                  drone_model,
                                                                                                  "Arrival")

        all_avg_col_percent_depart[i] = average_percent_collision_d
        all_avg_time_depart[i] = average_time_d
        all_avg_altitude_depart[i] = average_altitude_d

        all_avg_col_percent_arrive[i] = average_percent_collision_a
        all_avg_time_arrive[i] = average_time_a
        all_avg_altitude_arrive[i] = average_altitude_a

    discretised_distance = np.linspace(min(distance_from_airport), max(distance_from_airport), 1000)
    # Plotting Average Collision percent against distance

    # spline = UnivariateSpline(distance_from_airport,all_avg_col_percent)
    avg_col_percent_spline_d = interpolate.make_interp_spline(distance_from_airport, all_avg_col_percent_depart)
    avg_col_percent_spline_a = interpolate.make_interp_spline(distance_from_airport, all_avg_col_percent_arrive)
    plt.figure()
    plt.plot(discretised_distance, avg_col_percent_spline_d(discretised_distance), '-b', label="Depart")
    plt.plot(distance_from_airport, all_avg_col_percent_depart, 'bo')
    plt.plot(discretised_distance, avg_col_percent_spline_a(discretised_distance), '-r', label="Arrival")
    plt.plot(distance_from_airport, all_avg_col_percent_arrive, 'ro')
    plt.title(
        "Average Collision Percentage against \n Drone distance from Airport Center - " + airport + " - " + drone_model)
    plt.xlabel("Drone distance from center of airport / km")
    plt.ylabel("Average collision percentage / %")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # THIS IS TO ELIMINATE TIMES WHERE THERE WERE NO COLLISIONS AT ALL IN THE SIMULATIONS
    if np.isnan(all_avg_time_arrive).any() or np.isnan(all_avg_time_depart).any():
        valid_indices_a = np.logical_not(np.isnan(all_avg_time_arrive))
        valid_indices_d = np.logical_not(np.isnan(all_avg_time_depart))

        distance_from_airport_a = np.array(distance_from_airport)[valid_indices_a]
        distance_from_airport_d = np.array(distance_from_airport)[valid_indices_d]

        all_avg_time_arrive = np.array(all_avg_time_arrive)[valid_indices_a]
        all_avg_time_depart = np.array(all_avg_time_depart)[valid_indices_d]

        all_avg_altitude_arrive = np.array(all_avg_altitude_arrive)[valid_indices_a]
        all_avg_altitude_depart = np.array(all_avg_altitude_depart)[valid_indices_d]

        discretised_distance_a = np.linspace(min(distance_from_airport_a), max(distance_from_airport_a), 1000)
        discretised_distance_d = np.linspace(min(distance_from_airport_d), max(distance_from_airport_d), 1000)

    else:
        discretised_distance_a = np.linspace(min(distance_from_airport), max(distance_from_airport), 1000)
        discretised_distance_d = np.linspace(min(distance_from_airport), max(distance_from_airport), 1000)
        distance_from_airport_d = distance_from_airport
        distance_from_airport_a = distance_from_airport

    # Plotting change in time of collision
    avg_time_fit_d = np.poly1d(np.polyfit(distance_from_airport_d, all_avg_time_depart, 3))
    avg_time_fit_a = np.poly1d(np.polyfit(distance_from_airport_a, all_avg_time_arrive, 3))
    plt.figure()
    plt.plot(discretised_distance_d, avg_time_fit_d(discretised_distance_d), '-b', label="Depart")
    plt.plot(distance_from_airport_d, all_avg_time_depart, 'bo')
    plt.plot(discretised_distance_a, avg_time_fit_a(discretised_distance_a), '-r', label="Arrival")
    plt.plot(distance_from_airport_a, all_avg_time_arrive, 'ro')
    plt.title("Average Collision Time against \n Drone distance from Airport Center - " + airport + " - " + drone_model)
    plt.xlabel("Drone distance from center of airport / km")
    plt.ylabel("Average collision time / s")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting average altitude of collisions against distance
    avg_altitude_fit_d = np.poly1d(np.polyfit(distance_from_airport_d, all_avg_altitude_depart, 3))
    avg_altitude_fit_a = np.poly1d(np.polyfit(distance_from_airport_a, all_avg_altitude_arrive, 3))
    # avg_altitude_fit_d = UnivariateSpline(distance_from_airport_d,all_avg_altitude_depart)
    # avg_altitude_fit_a = UnivariateSpline(distance_from_airport_a, all_avg_altitude_arrive)
    plt.figure()
    plt.plot(discretised_distance_d, avg_altitude_fit_d(discretised_distance_d), '-b', label="Depart")
    plt.plot(distance_from_airport_d, all_avg_altitude_depart, 'bo')
    plt.plot(discretised_distance_a, avg_altitude_fit_a(discretised_distance_a), '-r', label="Arrival")
    plt.plot(distance_from_airport_a, all_avg_altitude_arrive, 'ro')
    plt.title(
        "Average Collision Altitude against \n Drone distance from Airport Center - " + airport + " - " + drone_model)
    plt.xlabel("Drone distance from center of airport / km")
    plt.ylabel("Average collision altitude / ft")
    plt.legend()
    plt.tight_layout()
    plt.show()


def DBSCAN_Clustering(eps,min_sample,airport,distance_from_airport,drone_model,depart_arrive):
    heathrow = [51.471305, -0.460861]
    gatwick = [51.153871, -0.187176]

    bounds_LHR = boundary(heathrow, 6)
    bounds_LGW = boundary(gatwick, 6)

    p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')

    if airport == "LHR":
        bounds = bounds_LHR
        airport_code = "EGLL"

    if airport == "LGW":
        bounds = bounds_LGW
        airport_code = "EGKK"

    _, _, _, drone_data, _, sim_index,_ = data_access(airport,distance_from_airport,drone_model,depart_arrive)

    # Converting back to m from feet for the sake of DBSCAN
    drone_data['altitude'] = drone_data['altitude'] * 0.3048

    drone_long = list(drone_data['longitude'])
    drone_lat = list(drone_data['latitude'])
    drone_altitude = list(drone_data['altitude'])

    # Converting back from degrees to m for the sake of DBSCAN so that all axis are in the same units
    drone_data['longitude'], drone_data['latitude'] = p(drone_long, drone_lat)


    clustering_data = drone_data[["longitude","latitude","altitude"]]
    db = DBSCAN(eps=eps, min_samples=min_sample).fit(clustering_data)

    drone_long = list(drone_data['longitude'])
    drone_lat = list(drone_data['latitude'])
    drone_altitude = list(drone_data['altitude'])

    drone_data['longitude'], drone_data['latitude'] = p(drone_long, drone_lat, inverse=True)



    drone_data['altitude'] = drone_data['altitude'] * 3.28084

    # Plotting Clusters of collisions
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.scatter(drone_data['longitude'], drone_data['latitude'], drone_data['altitude'], c=db.labels_)
    plt.title("Clusters of Collisions between Aircraft and Drone " + airport + " " + depart_arrive + "\nClusters: " + str(
        len(Counter(db.labels_)) - 1))
    ax1.set_xlabel("Longitude / $\circ$")
    ax1.set_ylabel('Latitude / $\circ$')
    ax1.set_zlabel('Altitude / feet')
    ax1.ticklabel_format(useOffset=False)
    plt.show()

    fig, ax = plt.subplots(
        1, figsize=(10, 10),
        subplot_kw=dict(projection=OSGB())
    )
    airports[airport_code].plot(ax)
    plt.scatter(drone_data['longitude'], drone_data['latitude'],color='red', transform=ccrs.PlateCarree())
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("Areas of Collisions above " + airport + " - " + depart_arrive)
    ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]))
    ax.title.set_fontsize(20)
    plt.show()

    drone_data.drop(drone_data.index[clustering_data.index[db.labels_ == -1]],inplace=True)

    # Figure to show the change in points from the anomaly detection
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.scatter(drone_data['longitude'], drone_data['latitude'], drone_data['altitude'], 'o', color='b')
    plt.title("Clusters of Collisions between Aircraft and Drone - " + airport + " " + depart_arrive + "\n without anomalies")
    ax1.set_xlabel("Longitude / $\circ$")
    ax1.set_ylabel('Latitude / $\circ$')
    ax1.set_zlabel('Altitude / feet')
    ax1.ticklabel_format(useOffset=False)
    plt.show()

    fig, ax = plt.subplots(
        1, figsize=(10, 10),
        subplot_kw=dict(projection=OSGB())
    )
    airports[airport_code].plot(ax)
    plt.scatter(drone_data['longitude'], drone_data['latitude'],color='red', transform=ccrs.PlateCarree())
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("Areas of Collisions above " + airport + " - " + depart_arrive)
    ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]))
    ax.title.set_fontsize(20)
    plt.show()

def sim_run_comparison(airport,distance_from_airport,drone_model,depart_arrive):
    _, _, _, drone_data, _, sim_index, _ = data_access(airport, distance_from_airport, drone_model, depart_arrive)

    # Plotting Airspeed comparison
    fig = plt.figure()
    plot_label = 1
    plt.plot(drone_data['drone_speed'].iloc[0:sim_index[0]], drone_data['aircraft_speed'].iloc[0:sim_index[0]], 'x',
             label=plot_label)
    for i in range(0, len(sim_index) - 1):
        plot_label += 1
        plt.plot(drone_data['drone_speed'].iloc[sim_index[i]:sim_index[i + 1]],
                 drone_data['aircraft_speed'].iloc[sim_index[i]:sim_index[i + 1]], 'x', label=plot_label)
    plot_label += 1
    plt.plot(drone_data['drone_speed'].iloc[sim_index[-1]:len(drone_data['drone_speed'])],
             drone_data['aircraft_speed'].iloc[sim_index[-1]:len(drone_data['drone_speed'])], 'x', label=plot_label)
    plt.title("Aircraft Groundspeed vs Drone Groundspeed")
    plt.xlabel("Drone Groundspeed / m/s")
    plt.ylabel("Aircraft Groundspeed / m/s")
    plt.legend(title='Run No.')
    plt.show()





airport = "LGW"
distance_from_airport = "5.0"
drone_model = 'Mini_2' # Either 'Mavic_3' or 'Mini_2'
depart_arrive = "Depart" # Either "Depart" or "Arrival"

#sim_run_comparison(airport,distance_from_airport,drone_model,depart_arrive)

#DBSCAN_Clustering(50,25,"LHR",distance_from_airport,drone_model,"Depart")
#DBSCAN_Clustering(50,25,"LHR",distance_from_airport,drone_model,"Arrival")

#distance_comparison(airport,drone_model,depart_arrive)
#distance_comparison("LHR",drone_model,"Arrival")

depart_arrival_comparison(airport,drone_model)

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


