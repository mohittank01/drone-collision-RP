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
    int track;


    string FilePath;
    int TotalCols;
    vector<string> AllData;
    int AllData_size;
    vector<int> Aircraft_Index;
    vector<string> Single_Aircraft;
    int Single_Aircraft_size;

    void CSVData();
    void AircraftIndex();
    void SingleAircraft(int index);
    void ColumnSelect(int column_no, double* column_pointer);
    void Takeoff_Time();
    void Arrive_Time();

    void PrintAircraft();

    public:
    void Set_Parameters_and_Data(string File_Path, int no_cols);
    void Vector_Allocation(int index_input);
    void Deallocation();
    int Vector_length;
    int takeoff_t;
    int arrive_t;

    int Aircraft_Index_size;

    double* longitude_vector;
    double* latitude_vector;
    double* altitude_vector;
    double* groundspeed_vector;
    double* track_vector;

};


#endif
