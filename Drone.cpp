#include "Drone.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <omp.h>

using namespace std;

void Drone::SetInitialParameters(string FilePath_input, int Vector_length_input, int TotalCols_input, int drone_index_input, int aircraft_index_input, int takeoff_time_input, double* air_long, double* air_lat, double* air_alt, double aircraft_radius_input, double drone_radius_input){
    FilePath = FilePath_input;
    VectorLength = Vector_length_input;
    TotalCols = TotalCols_input;
    DroneIndex = drone_index_input;
    AircraftIndex = aircraft_index_input;
    TakeoffTime = takeoff_time_input;
    AircraftRadius = aircraft_radius_input;
    DroneRadius = drone_radius_input;

    // Column numbers
    longitude_col_no = 1;
    latitude_col_no = 2;
    heading_col_no = 3;

    // Obtain inital position data from Python csv file
    CSVData();

    initial_long_pos = new double[1];
    initial_lat_pos = new double[1];
    initial_heading = new double[1];

    collision_index = new int[1];


    *initial_long_pos = stod(PositionData[((DroneIndex+1)*TotalCols) + longitude_col_no]);
    *initial_lat_pos = stod(PositionData[((DroneIndex+1)*TotalCols) + latitude_col_no]);
    *initial_heading = stod(PositionData[((DroneIndex+1)*TotalCols) + heading_col_no]);

    aircraft_longitude = new double[VectorLength];
    aircraft_latitude = new double[VectorLength];
    aircraft_altitude = new double[VectorLength];

    for(int i = 0; i < VectorLength; ++i){
        aircraft_longitude[i] = air_long[i];
        aircraft_latitude[i] = air_lat[i];
        aircraft_altitude[i] = air_alt[i];
    }

    if(aircraft_altitude[0] != 0){
        depart_or_arrive = 1; // ARRIVAL 
    }
    else{
        depart_or_arrive = 0; // DEPART
    }
 
    start_alt = 100.0; // Starting Altitude
    max_straight_speed = 19.0;
    max_ascend_speed = 8.0;
}

void Drone::ClearOutput(int Aircraft_Index, string distance_from_airport){
    string aircraft_index = to_string(Aircraft_Index);
    ofstream outfile;
    outfile.open("Drone_Collisions_" + distance_from_airport + "_km/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::trunc);
    outfile.close();
}

void Drone::ClearOutput_1File(string distance_from_airport){
    ofstream outfile;
    outfile.open("Drone_Collisions_" + distance_from_airport + "_km/All_Drone_Collisions.csv", ofstream::out | ofstream::trunc);
    outfile.close();
}


void Drone::SetInitialConditions(){
    // Allocate memory to vectors
    longitude_vector = new double[VectorLength];
    latitude_vector = new double[VectorLength];
    altitude_vector = new double[VectorLength];
    speed_vector = new double[VectorLength];
    heading_vector = new double[VectorLength];

    // Initialise vectors
    longitude_vector[0] = *initial_long_pos; // Initialising longitude
    latitude_vector[0] = *initial_lat_pos; // Initialising latitude
    heading_vector[0] = *initial_heading; // Initialising heading
    altitude_vector[0] = start_alt; // 100m starting altitude
    speed_vector[0] = max_straight_speed; // 19m/s max strightline speed
}


void Drone::CSVData(){
    ifstream csv;

    csv.open(FilePath);

    if (csv.fail()) {
        cout << "ERROR - Failed to open " << FilePath << endl;
    }

    string line;
    string element;
    
    // Seperating line by line and then element by element 
    while(getline(csv, line)){
        stringstream new_line(line);
        while (getline(new_line, element, ',')){
        PositionData.push_back(element);
        }
    }
    csv.close();

    PositionData_size = PositionData.size();
}

void Drone::FirstStage(){
    int min_t_1st = 1;
    int max_t_1st = TakeoffTime;

    //random_device seed;d
    //mt19937 engine(seed());
    default_random_engine engine{random_device{}()};
    uniform_int_distribution<int> dist(min_t_1st, max_t_1st); // uniform, unbiased

    random_t_1st = dist(engine);
    //random_t_1st = rand()%(max_t_1st-min_t_1st + 1) + min_t_1st;


    for(int i = 1; i < random_t_1st; ++i){
        longitude_vector[i] = longitude_vector[i-1] + sin(heading_vector[i-1])*max_straight_speed;
        latitude_vector[i] = latitude_vector[i-1] + cos(heading_vector[i-1])*max_straight_speed;
        heading_vector[i] = heading_vector[i-1];
        speed_vector[i] = max_straight_speed;
        altitude_vector[i] = start_alt;
    }
}

void Drone::CubedVolume(){
    // LHR
    min_cube_alt = 100;
    max_cube_alt = min_cube_alt + 400; // ADDING 400m TO THE MIN ALTITUDE

    // 27R
    if(aircraft_latitude[0] <= 5706340.62 && aircraft_latitude[0] >= 5705792.58){
        min_cube_lat = 5705770.46;
        max_cube_lat = min_cube_lat + 500; // ADDING 500m TO THE MIN FOR THE RUNWAY WIDTH
        if(depart_or_arrive){ // ARRIVAL
            min_cube_long = 676345.60;
            max_cube_long = min_cube_long + 5000;
        }
        else{ // DEPARTURE
            max_cube_long = 676819.02; 
            min_cube_long = max_cube_long - 5000; // FOR THE LENGTH OF THE CUBED VOLUME - 5000m
        }
    }
    // 27L
    else if(aircraft_latitude[0] <= 5705065.74 && aircraft_latitude[0] >= 5704388.38){
        max_cube_lat = 5704800.46;
        min_cube_lat = max_cube_lat - 500; // MINUS 500m TO THE MAX FOR THE RUNWAY WIDTH

        if(depart_or_arrive == 1){ // ARRIVAL
            min_cube_long = 676345.60;
            max_cube_long = min_cube_long + 5000;
        }
        else{ // DEPARTURE
            max_cube_long = 676819.02; 
            min_cube_long = max_cube_long - 5000; // FOR THE LENGTH OF THE CUBED VOLUME - 5000m
        }
    }
}


void Drone::SecondStage(){
    CubedVolume();

    // Seed generator
    //random_device seed;
    //mt19937 engine(seed());
    default_random_engine engine_1{random_device{}()};
    default_random_engine engine_2{random_device{}()};
    default_random_engine engine_3{random_device{}()};
    
    // Uniform distribution - LONGITUDE
    uniform_real_distribution<double> long_dist(min_cube_long, max_cube_long); // uniform, unbiased

    // Uniform distribution - LATITUDE
    uniform_real_distribution<double> lat_dist(min_cube_lat, max_cube_lat); // uniform, unbiased

    // Uniform distribution - ALTITUDE
    uniform_real_distribution<double> alt_dist(min_cube_alt, max_cube_alt); // uniform, unbiased

    random_long = long_dist(engine_1);
    //random_long = (rand() / (double)RAND_MAX) * (max_cube_long-min_cube_long) + min_cube_long;
    random_lat = lat_dist(engine_2);
    //random_lat = (rand() / (double)RAND_MAX) * (max_cube_lat-min_cube_lat) + min_cube_lat;
    random_alt = alt_dist(engine_3);
    //random_alt = (rand() / (double)RAND_MAX) * (max_cube_alt-min_cube_alt) + min_cube_alt;

    double pitch_angle;
    double modulus_long_lat;

    int last_index = random_t_1st - 1;

    double long_diff = abs(random_long - longitude_vector[last_index]);
    double lat_diff = abs(random_lat - latitude_vector[last_index]);
    double alt_diff = abs(random_alt - altitude_vector[last_index]);

    // TOP LEFT
    if(random_long <= longitude_vector[last_index] && random_lat >= latitude_vector[last_index]){
        heading_angle = 2*M_PI - atan(long_diff / lat_diff);
    }
    // BOTTOM LEFT
    else if(random_long <= longitude_vector[last_index] && random_lat <= latitude_vector[last_index]){
        heading_angle = 1.5*M_PI - atan(lat_diff / long_diff);
    }
    // BOTTOM RIGHT
    else if(random_long >= longitude_vector[last_index] && random_lat <= latitude_vector[last_index]){
        heading_angle = M_PI - atan(long_diff / lat_diff);
    }
    // TOP RIGHT
    else if(random_long >= longitude_vector[last_index] && random_lat >= latitude_vector[last_index]){
        heading_angle = atan(long_diff / lat_diff);
    }
    
    modulus_long_lat = sqrt(long_diff*long_diff + lat_diff*lat_diff);

    pitch_angle = atan(alt_diff / modulus_long_lat);

    double velocity_factor = (max_straight_speed - (2*(max_straight_speed - max_ascend_speed)/M_PI)*pitch_angle);



    for(int i = random_t_1st; i < VectorLength; ++i){
        longitude_vector[i] = longitude_vector[i-1] + sin(heading_angle) * velocity_factor;
        latitude_vector[i] = latitude_vector[i-1] + cos(heading_angle) * velocity_factor;
        altitude_vector[i] = altitude_vector[i-1] + sin(pitch_angle) * velocity_factor;
        speed_vector[i] = velocity_factor;
        heading_vector[i] = heading_angle;
    }
}

void Drone::Output_1File(int run_no, string distance_from_airport){
    string aircraft_index = to_string(AircraftIndex);
    ofstream outfile1;
    outfile1.open("Drone_Collisions_" + distance_from_airport + "_km/All_Drone_Collisions.csv", ofstream::out | ofstream::app);
    outfile1.precision(10);
    for (int i=0; i < VectorLength; ++i){
        if(i == *collision_index){
            outfile1 << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << AircraftIndex << "," << 1 << '\n';    
        }
        else{
            outfile1 << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << AircraftIndex << "," << 0 << '\n';
        }
    }
    outfile1 << '\n';
    outfile1.close();
}

void Drone::Output(int run_no, string distance_from_airport){
    string aircraft_index = to_string(AircraftIndex);
    ofstream outfile;
    outfile.open("Drone_Collisions_" + distance_from_airport + "_km/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::app);
    outfile.precision(10);
    for (int i=0; i < VectorLength; ++i){
        if(i == *collision_index){
            outfile << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << 1 << '\n';    
        }
        else{
            outfile << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << 0 << '\n';
        }
    }
    outfile << '\n';
    outfile.close();
}

void Drone::Output_Collision_Num(double* local_collisions, string distance_from_airport){
    string aircraft_index = to_string(AircraftIndex);
    ofstream outfile;
    outfile.open("Drone_Collisions_" + distance_from_airport + "_km/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::app);
    outfile << *local_collisions;
}

void Drone::Output_1File_Collision_Num(double* total_collisions, string distance_from_airport){
    ofstream outfile1;
    outfile1.open("Drone_Collisions_" + distance_from_airport + "_km/All_Drone_Collisions.csv", ofstream::out | ofstream::app);
    outfile1 << *total_collisions;
}

void Drone::Deallocate(){
    delete[] longitude_vector;
    delete[] latitude_vector;
    delete[] altitude_vector;
    delete[] speed_vector;
    delete[] heading_vector;
    delete[] collision_index;
}

bool Drone::Collision(){
    for(int i = 0; i < VectorLength; ++i){
        double distance = sqrt(
            (longitude_vector[i] - aircraft_longitude[i]) * (longitude_vector[i] - aircraft_longitude[i]) +
            (latitude_vector[i] - aircraft_latitude[i]) * (latitude_vector[i] - aircraft_latitude[i]) + 
            (altitude_vector[i] - aircraft_altitude[i]) * (altitude_vector[i] - aircraft_altitude[i])
        );
        if (distance < (DroneRadius + AircraftRadius)){
            *collision_index = i;
            return 1;
        }
    }
    return 0;
}



void Drone::Simulation(int number_runs, double* total_collisions, double* local_collisions, string distance_from_airport){
    SetInitialConditions();
    int i;
    //#pragma omp parallel for private(i) firstprivate(longitude_vector, latitude_vector, altitude_vector, heading_vector, speed_vector) shared(total_collisions)
    for(i = 0; i < number_runs; ++i){
        FirstStage();   // Drone heads towards centre of runway
        SecondStage();  // Drone heads towards random coords in volume
        if (Collision()){
            Output(i, distance_from_airport);   // Outputs to csv file for each aircraft index
            Output_1File(i, distance_from_airport); // Outputs to A SINGLE csv file for all collisions 
            *local_collisions += 1;
            *total_collisions += 1;
        }
    }
    Deallocate();

}
