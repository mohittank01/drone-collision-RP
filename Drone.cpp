#include "Drone.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>

using namespace std;

void Drone::SetInitialConditions(string FilePath_input, int Vector_length_input, int TotalCols_input, int drone_index_input, int takeoff_time_input){
    FilePath = FilePath_input;
    VectorLength = Vector_length_input;
    TotalCols = TotalCols_input;
    DroneIndex = drone_index_input;
    TakeoffTime = takeoff_time_input;

    // Column numbers
    longitude_col_no = 1;
    latitude_col_no = 2;
    heading_col_no = 3;

    // Obtain inital position data from Python csv file
    CSVData();

    // Allocate memory to vectors
    longitude_vector = new double[VectorLength];
    latitude_vector = new double[VectorLength];
    altitude_vector = new double[VectorLength];
    speed_vector = new double[VectorLength];
    heading_vector = new double[VectorLength];

    longitude_factor_1st = new int[1];
    latitude_factor_1st = new int[1];

    // Initialise vectors
    longitude_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + longitude_col_no]); // Initialising longitude 
    latitude_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + latitude_col_no]); // Initialising latitude
    heading_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + heading_col_no]); // Initialising heading
    altitude_vector[0] = 100; // 100m starting altitude
    speed_vector[0] = 19; // 19m/s max strightline speed

    max_straight_speed = 19.0;

    FirstStage();
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

// DONT ACTUIALLY NEED THIS FUCNTION BUT KEPT IT JUST IN CASE DOUBT ID NEED IT
void Drone::HeadingCalc(double heading_val, int* longitude_factor, int* latitude_factor){
    // Top right quadrant
    if(heading_val >= M_PI && heading_val <= 1.5*M_PI){
        *longitude_factor = -1;
        *latitude_factor = -1;
    }
    // Bottom right quadrant
    else if(heading_val >= 1.5*M_PI && heading_val <= 2*M_PI){
        *longitude_factor = -1;
        *latitude_factor = 1;
    }
    // Bottom left quadrant
    else if(heading_val >= 0 && heading_val <= 0.5*M_PI){
        *longitude_factor = 1;
        *latitude_factor = 1;
    }
    // Top left quadrant
    else if(heading_val >= 0.5*M_PI && heading_val <= M_PI){
        *longitude_factor = 1;
        *latitude_factor = -1;
    }
}


void Drone::FirstStage(){
    int min_t_1st = 0;
    int max_t_1st = TakeoffTime;

    random_device seed;
    mt19937 engine(seed());
    uniform_int_distribution<int> dist(min_t_1st, max_t_1st); // uniform, unbiased

    int random_t_1st = dist(engine);
    
    for(int i = 1; i <= random_t_1st; ++i){
        longitude_vector[i] = longitude_vector[i-1] + sin(heading_vector[i-1])*max_straight_speed;
        latitude_vector[i] = latitude_vector[i-1] + cos(heading_vector[i-1])*max_straight_speed;
        heading_vector[i] = heading_vector[i-1];
        speed_vector[i] = max_straight_speed;
    }



}