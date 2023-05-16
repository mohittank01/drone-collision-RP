#include "Aircraft.h"
#include "Drone.h"
#include <iostream>
#include <omp.h>
#include <boost/program_options.hpp>
#include <iomanip>
#include <sstream>

namespace po = boost::program_options;
using namespace std;

int main(int argc, char* argv[]){

  po::options_description opts("Available options.");
  opts.add_options()
    ("airport", po::value<string>()->default_value("LHR"), "Airport Code.")
    ("drone_model", po::value<string>()->default_value("Mavic_3"), "Drone Model: 'Mavic_3' or 'Mini_2'.")
    ("depart_arrive", po::value<int>()->default_value(1), "Departure (0) or Arrival (1). ")
    ("distance", po::value<double>()->default_value(1.0), "Inital distance of drones from centre of airport.")
    ("total_runs", po::value<int>()->default_value(1), "Total run number of program.")
    ("help", "Print help message.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if(vm.count("help")){
    cout << opts << endl;
    return 1;
  }
  
  stringstream stream;
  stream << fixed << setprecision(1) << vm["distance"].as<double>();

  string Airport = vm["airport"].as<string>();
  string DroneModel = vm["drone_model"].as<string>();
  int depart_or_arrive = vm["depart_arrive"].as<int>();
  string distance_from_airport = stream.str();
  int max_run_number = vm["total_runs"].as<int>();

  string FilePath_aircraft;

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
  int no_col_aircraft = 22; // Total number of  Columns in Aircraft CSV file
  double aircraft_radius = 5.75; // Frontal Area of aircraft (A320ceo) - distance from engine to centre line 

  string FilePath_drone = Airport + "/drone_inital_positions_" + distance_from_airport + "km.csv"; /// THIS CHANGES EVERYTHING
  int no_col_drone = 4; // Total number of columns in Drone CSV file
  int drone_total = 99; // Total number of drones around airport


  double* total_collisions = new double[1];
  *total_collisions = 0;

  double* total_sims = new double[1];
  *total_sims = 0;

  Aircraft Aircraft;
  Drone Drone;

  Aircraft.Set_Parameters_and_Data(FilePath_aircraft, no_col_aircraft);
  Drone.Average_ClearOutput_1File(distance_from_airport, Airport, DroneModel, depart_or_arrive);
  Drone.CSVData(FilePath_drone);

  for(int run_number=0; run_number < max_run_number; ++run_number){
    int i;
    double* local_collisions = new double[1];
    *local_collisions = 0;
    #pragma omp parallel for private(i) firstprivate(Aircraft,Drone)
    for(i=0; i < Aircraft.Aircraft_Index_size-1; ++i){
      Aircraft.Vector_Allocation(i);
      for(int j=0; j < drone_total; ++j){
        Drone.SetInitialParameters(Airport, DroneModel, Aircraft.Vector_length, no_col_drone, j, i, Aircraft.takeoff_t, Aircraft.arrive_t, Aircraft.longitude_vector, Aircraft.latitude_vector, Aircraft.altitude_vector, Aircraft.track_vector, Aircraft.groundspeed_vector, aircraft_radius);
        Drone.Simulation(10000, total_collisions, local_collisions, distance_from_airport, run_number, total_sims);
      }
      Aircraft.Deallocation();
    }
    Drone.AverageOutputFile_LocalCollision(Airport, DroneModel, local_collisions, distance_from_airport, run_number, depart_or_arrive);
    cout << "Number of collisions: " << *local_collisions << endl;
  }
  Drone.AverageOutputFile_TotalCollision(Airport, DroneModel, total_collisions, distance_from_airport, total_sims, max_run_number, depart_or_arrive);
 }
