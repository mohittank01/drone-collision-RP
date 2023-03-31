#include "Aircraft.h"
#include <iostream>


using namespace std;

int main(){
  
  // Open csv file for DEPARTURES
  string FilePath = "TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv";
  int no_col = 22;
  int index = 0;

  Aircraft Test;

  Test.Set_Parameters_and_Data(FilePath, no_col, index);
  Test.Vector_Allocation();
  
 }
