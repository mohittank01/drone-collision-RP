#ifndef CLASS_AIRCRAFT
#define CLASS_AIRCRAFT
#include <iostream>
#include <vector>

using namespace std;


class Aircraft{
    private:

    int callsign;
    int longitude;
    int latitude;
    int altitude;
    int groundspeed;
    int onground;
    int index;


    string FilePath;
    int TotalCols;
    vector<string> AllData;
    int AllData_size;
    vector<int> Aircraft_Index;
    int Aircraft_Index_size;
    vector<string> Single_Aircraft;
    int Single_Aircraft_size;

    int takeoff_t;

    double* longitude_vector;
    double* latitude_vector;
    double* altitude_vector;
    double* groundspeed_vector;

    void CSVData();
    void AircraftIndex();
    void SingleAircraft();
    void ColumnSelect(int column_no, double* column_pointer);
    void Takeoff_Time();

    void PrintAircraft();

    public:
    void Set_Parameters_and_Data(string File_Path, int no_cols, int index_input);
    void Vector_Allocation();
    int Vector_length;

};


#endif
