#include "Aircraft.h"
#include "Drone.h"
#include <iostream>
#include <omp.h>


using namespace std;

int main(){

  string FilePath_aircraft;
  
  // VARIABLES TO CHANGE 
  string Airport = "LHR";
  int depart_or_arrive = 1; // 0 - DEPART 1 - ARRIVE
  string distance_from_airport = "1"; // DISTANCE FROM CENTRE OF RUNWAY

  if(Airport == "LHR"){
    if (!depart_or_arrive){ // DEPART
      FilePath_aircraft = Airport + "/TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv";
    }
    else{ // ARRIVE
      FilePath_aircraft = Airport + "/LANDING_UTM_ARRIVE_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv";
    } 
  }

  if(Airport == "LGW"){
    if(!depart_or_arrive){ // DEPART
      FilePath_aircraft = Airport + "/TAKEOFF_UTM_DEPART_A320 LGW time - 2017-08-25 04-00 to 2017-08-25 23-45.csv";
    }
    else{ // ARRIVE
      FilePath_aircraft = Airport + "/LANDING_UTM_ARRIVE_A320 LGW time - 2017-08-25 04-00 to 2017-08-25 23-45.csv";
    }
  }
  
  int no_col_aircraft = 22;
  double aircraft_radius = 3.5;

  string FilePath_drone = Airport + "/drone_inital_positions_" + distance_from_airport + "km.csv"; /// THIS CHANGES EVERYTHING
  int no_col_drone = 4;
  int drone_total = 99;
  double drone_radius = 0.19;

  double* total_collisions = new double[1];
  *total_collisions = 0;

  Aircraft Aircraft;
  Drone Drone;

  Aircraft.Set_Parameters_and_Data(FilePath_aircraft, no_col_aircraft);
  Drone.ClearOutput_1File(distance_from_airport, Airport, depart_or_arrive);
  int i;

  #pragma omp parallel for private(i) firstprivate(Aircraft,Drone)
  for(i=0; i < Aircraft.Aircraft_Index_size-1; ++i){
    Aircraft.Vector_Allocation(i);
    Drone.ClearOutput(i, distance_from_airport, Airport, depart_or_arrive);
    double* local_collisions = new double[1];
    *local_collisions = 0;
    for(int j=0; j < drone_total; ++j){
      Drone.SetInitialParameters(FilePath_drone, Airport, Aircraft.Vector_length, no_col_drone, j, i, Aircraft.takeoff_t, Aircraft.arrive_t, Aircraft.longitude_vector, Aircraft.latitude_vector, Aircraft.altitude_vector, Aircraft.track_vector, aircraft_radius, drone_radius);
      Drone.Simulation(10000, total_collisions, local_collisions, distance_from_airport);
    }
    Drone.Output_Collision_Num(local_collisions, distance_from_airport);
    Aircraft.Deallocation();

  }
  Drone.Output_1File_Collision_Num(total_collisions, distance_from_airport, Airport, depart_or_arrive);
  cout << "Number of collisions: " << *total_collisions << endl;
 }
