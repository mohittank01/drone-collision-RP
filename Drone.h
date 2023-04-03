#ifndef CLASS_DRONE
#define CLASS_DRONE

#include <iostream>
#include <vector>


using namespace std;

class Drone{
    private:

    int VectorLength;
    int DroneIndex;
    int TakeoffTime;
    int longitude_col_no;
    int latitude_col_no;
    int heading_col_no;

    int PositionData_size;
    int TotalCols;
    vector<string> PositionData;
    string FilePath;

    int* longitude_factor_1st;
    int* latitude_factor_1st;

    void CSVData();
    void ColumnSelect_Index(int column_no, double* column_pointer, int DroneIndex);
    void HeadingCalc(double heading_val, int* longitude_vector, int* latitude_vector);
    void FirstStage();

    double* longitude_vector;
    double* latitude_vector;
    double* altitude_vector;
    double* speed_vector;
    double* heading_vector;

    double max_straight_speed;

    public:

    void SetInitialConditions(string FilePath_input, int Vector_length_input, int TotalCols_input, int drone_index_input, int takeoff_time_input);



};

#endif
