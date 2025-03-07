#include "Aircraft.h"
#include "Drone.h"
#include <iostream>


using namespace std;

int main(){
  
  // Open csv file for DEPARTURES
  string FilePath_aircraft = "TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv";
  int no_col_aircraft = 22;
  double aircraft_radius = 3.5;

  string FilePath_drone = "drone_inital_positions_1km.csv"; /// THIS CHANGES EVERYTHING
  int no_col_drone = 4;
  int drone_total = 99;
  double drone_radius = 0.19;

  double* total_collisions = new double[1];
  *total_collisions = 0;

  Aircraft Aircraft;
  Drone Drone;

  Aircraft.Set_Parameters_and_Data(FilePath_aircraft, no_col_aircraft);

  for(int i=0; i < 1; ++i){
    Aircraft.Vector_Allocation(i);
    Drone.ClearOutput(i);
    for(int j=0; j < drone_total; ++j){
      Drone.SetInitialParameters(FilePath_drone, Aircraft.Vector_length, no_col_drone, j, i, Aircraft.takeoff_t, Aircraft.longitude_vector, Aircraft.latitude_vector, Aircraft.altitude_vector, aircraft_radius, drone_radius);
      Drone.Simulation(10000, total_collisions);
    }
    Aircraft.Deallocation(); // PLAY AROUND WITH THIS LINE

  }
  cout << "Number of collisions: " << *total_collisions << endl;
 }
