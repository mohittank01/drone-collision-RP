#include "Drone.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace std;

void Drone::SetInitialConditions(string FilePath_input, int Vector_length_input, int TotalCols_input, int drone_index_input){
    FilePath = FilePath_input;
    VectorLength = Vector_length_input;
    TotalCols = TotalCols_input;
    DroneIndex = drone_index_input;

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

    // Initialise vectors
    longitude_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + longitude_col_no]); // Initialising longitude 
    latitude_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + latitude_col_no]); // Initialising latitude
    heading_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + heading_col_no]); // Initialising heading
    altitude_vector[0] = 100; // 100m starting altitude
    speed_vector[0] = 19; // 19m/s max strightline speed


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
