#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
using namespace std;

void PrintAircraft(vector<string> aircraft, int ldh){
  cout.precision(4);
  for (int i = 0; i < aircraft.size() / ldh ; ++i){
    for (int j = 0; j < ldh; ++j){
      cout << setw(4) << aircraft[i*ldh + j] << " ";
    }
    cout << endl;
  }
}

vector<int> AircraftIndex(vector<string> main_data, int ldh){
  // Callsign column number
  int callsign = 3;
  vector<int> aircraft_index;
  for(int i = 0; i < main_data.size() / ldh; ++i){
    if ((i+1)*ldh + callsign >= main_data.size()) {
        // The next aircraft does not exist, so break out of the loop
        break;
    }
    if (main_data[i*ldh + callsign] != main_data[(i+1)*ldh + callsign]) {
        aircraft_index.push_back((i+1)*ldh);    
    }
  }
  return aircraft_index;
}


vector<string> SingleAircraft(vector<string> main_data, vector<int> aircraft_index, int index){
  vector<string> single_aircraft;
  if (index == aircraft_index.size()){
    for (int i = aircraft_index[index-1]; i < main_data.size(); ++i){
      single_aircraft.push_back(main_data[i]);
    }
    return single_aircraft;
  }
  
  for (int i = aircraft_index[index]; i < aircraft_index[index+1]; ++i){
    single_aircraft.push_back(main_data[i]);
  }
  
  return single_aircraft;
}

vector<string> CSVData(string FilePath, int ldh){
  ifstream csv;

  csv.open(FilePath);

  if (csv.fail()) {
    cout << "ERROR - Failed to open " << FilePath << endl;
  }

  vector<string> data;
  string line;
  string element;
  
  // Seperating line by line and then element by element 
  while(getline(csv, line)){
    stringstream new_line(line);
    while (getline(new_line, element, ',')){
      data.push_back(element);
    }
  }
  csv.close();

  return data;  
}

vector<double> ColumnSelect(vector<string> aircraft, int column_no, int ldh){
  vector<double> column;
  
  for(int i = 0; i < aircraft.size() / ldh; ++i){
    column.push_back(stod(aircraft[(i*ldh) + column_no]));
  }
  
  return column;
}


int main(){
  
  // Open csv file for DEPARTURES
  string FilePath = "TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv";
  
  vector<string> data;
  vector<string> single_aircraft;
  vector<int> aircraft_index;
  vector<double> longitude_vector;
  string line;
  string element;

  // Number of Columns
  int ldh = 22;
  // Callsign column number
  int callsign = 3;
  // Longitude column number
  int longitude = 14;
  // Latitude column number
  int latitude = 13;
  // Altitude column number 
  int altitude = 2;
  // Groundspeed column number
  int groundspeed = 8;

  // Obtaining data from csv data FilePath
  data = CSVData(FilePath,ldh);
  
  // Finds the index in the data array for the aircraft indexes
  aircraft_index = AircraftIndex(data, ldh);

  // Identify a single aircraft 
  single_aircraft = SingleAircraft(data, aircraft_index, 0);
  
  // Identify longitude vector as a test
  longitude_vector = ColumnSelect(single_aircraft, longitude, ldh);
 }
