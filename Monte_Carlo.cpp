#include "Aircraft.h"
#include "Drone.h"
#include <iostream>


using namespace std;

int main(){
  
  // Open csv file for DEPARTURES
  string FilePath_aircraft = "TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv";
  int no_col_aircraft = 22;
  int index_aircraft = 0;
  double aircraft_radius = 3.5;

  string FilePath_drone = "drone_inital_positions_5km.csv";
  int no_col_drone = 4;
  int index_drone = 75;
  double drone_radius = 0.19;

  Aircraft Aircraft;
  Drone Drone;

  Aircraft.Set_Parameters_and_Data(FilePath_aircraft, no_col_aircraft, index_aircraft);
  Aircraft.Vector_Allocation();


  Drone.SetInitialParameters(FilePath_drone, Aircraft.Vector_length, no_col_drone, index_drone, index_aircraft, Aircraft.takeoff_t, Aircraft.longitude_vector, Aircraft.latitude_vector, Aircraft.altitude_vector, aircraft_radius, drone_radius);
  Drone.Simulation(30);
 }
