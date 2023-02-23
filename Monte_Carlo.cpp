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

int main(){
  
  // Open csv file for DEPARTURES
  string FilePath = "TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv";
  ifstream departures;

  departures.open(FilePath);

  // TESTING IF FILE IS OPENED CORRECTLY
  if (departures.fail()) {
    cout << "ERROR - Failed to open " << FilePath << endl;
    return(0);
  }

  // Number of Columns
  int ldh = 22;
  // Callsign column number
  int callsign = 3;

  vector<string> data;
  vector<string> single_aircraft;
  string line;
  string element;
  
  // Seperating line by line and then element by element 
  while(getline(departures, line)){
    stringstream new_line(line);
    while (getline(new_line, element, ',')){
      data.push_back(element);
    }
  }
  
  // Loop to obtain a single aircraft 
  for (int i = 1 ; i < data.size() / ldh; ++i) {
    if ((i+1)*ldh + callsign >= data.size()) {
        // The next aircraft does not exist, so break out of the loop
        break;
    }
    for (int j = 0; j < ldh; ++j) {
        single_aircraft.push_back(data[i*ldh + j]);
    }

    if (data[i*ldh + callsign] != data[(i+1)*ldh + callsign]) {
            break;    
    }
  }

  PrintAircraft(single_aircraft, ldh);
  
  departures.close();

 }
