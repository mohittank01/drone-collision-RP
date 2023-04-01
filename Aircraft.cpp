#include "Aircraft.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

using namespace std;

void Aircraft::Set_Parameters_and_Data(string File_Path, int no_cols, int index_input){
    FilePath = File_Path;
    TotalCols = no_cols;
    // Callsign column number
    callsign = 3;
    // Longitude column number
    longitude = 14;
    // Latitude column number
    latitude = 13;
    // Altitude column number 
    altitude = 2;
    // Groundspeed column number
    groundspeed = 8;

    // Chosen Aircraft index
    index = index_input;

    CSVData();
    AircraftIndex();
}



void Aircraft::CSVData(){
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
        AllData.push_back(element);
        }
    }
    csv.close();

    AllData_size = AllData.size();
}


void Aircraft::AircraftIndex(){
    for(int i = 0; i < AllData_size / TotalCols; ++i){
        if ((i+1)*TotalCols + callsign >= AllData_size) {
            // The next aircraft does not exist, so break out of the loop
            break;
        }
        if (AllData[i*TotalCols + callsign] != AllData[(i+1)*TotalCols + callsign]) {
            Aircraft_Index.push_back((i+1)*TotalCols);    
        }
    }
    Aircraft_Index_size = Aircraft_Index.size();
}


void Aircraft::SingleAircraft(){
    if (index == Aircraft_Index_size){
        for (int i = Aircraft_Index[index-1]; i < AllData_size; ++i){
        Single_Aircraft.push_back(AllData[i]);
        }
    }

    else{
        for (int i = Aircraft_Index[index]; i < Aircraft_Index[index+1]; ++i){
            Single_Aircraft.push_back(AllData[i]);
        }
    }

    Single_Aircraft_size = Single_Aircraft.size();
}

void Aircraft::ColumnSelect(int column_no, double* column_pointer){
  vector<double> column;
  
  for(int i = 0; i < Single_Aircraft_size / TotalCols; ++i){
    column.push_back(stod(Single_Aircraft[(i*TotalCols) + column_no]));
  }

  int column_size = column.size();

  for (int i = 0; i < column_size; ++i){ 
    column_pointer [i] = column[i];
  }
  
}

void Aircraft::Vector_Allocation(){
    SingleAircraft();
    
    Vector_length = Single_Aircraft_size/TotalCols;

    longitude_vector = new double[Vector_length];
    latitude_vector = new double[Vector_length];
    altitude_vector = new double[Vector_length];
    groundspeed_vector = new double[Vector_length];

    ColumnSelect(longitude, longitude_vector);
    ColumnSelect(latitude, latitude_vector);
    ColumnSelect(altitude, altitude_vector);
    ColumnSelect(groundspeed, groundspeed_vector);

    cout.precision(10);
    cout << longitude_vector[10] << endl;
    cout << latitude_vector[10] << endl;
    cout << altitude_vector[10] << endl;

}



/*
void Aircraft::PrintAircraft(vector<string> aircraft, int ldh){
  cout.precision(4);
  for (int i = 0; i < aircraft.size() / ldh ; ++i){
    for (int j = 0; j < ldh; ++j){
      cout << setw(4) << aircraft[i*ldh + j] << " ";
    }
    cout << endl;
  }
}
*/