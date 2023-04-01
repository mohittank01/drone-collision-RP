#ifndef CLASS_DRONE
#define CLASS_DRONE

#include <iostream>
#include <vector>


using namespace std;

class Drone{
    private:

    int VectorLength;
    int DroneIndex;
    int longitude_col_no;
    int latitude_col_no;
    int heading_col_no;

    int PositionData_size;
    int TotalCols;
    vector<string> PositionData;
    string FilePath;

    void CSVData();
    void ColumnSelect_Index(int column_no, double* column_pointer, int DroneIndex);

    public:

    void SetInitialConditions(string FilePath_input, int Vector_length_input, int TotalCols_input, int drone_index_input);

    double* longitude_vector;
    double* latitude_vector;
    double* altitude_vector;
    double* speed_vector;
    double* heading_vector;


};

#endif
