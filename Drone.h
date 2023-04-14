#ifndef CLASS_DRONE
#define CLASS_DRONE

#include <iostream>
#include <vector>
#include "omp.h"


using namespace std;

class Drone{
    private:

    int VectorLength;
    int DroneIndex;
    int AircraftIndex;
    int TakeoffTime;

    double AircraftRadius;
    double DroneRadius;

    int longitude_col_no;
    int latitude_col_no;
    int heading_col_no;

    int PositionData_size;
    int TotalCols;
    vector<string> PositionData;
    string FilePath;

    double* initial_long_pos;
    double* initial_lat_pos;
    double* initial_heading;

    int* longitude_factor_1st;
    int* latitude_factor_1st;

    void CSVData();
    void ColumnSelect_Index(int column_no, double* column_pointer, int DroneIndex);
    void SetInitialConditions();
    void FirstStage();
    void CubedVolume();
    void SecondStage();
    bool Collision();
    void Output(int run_no);
    void Output_1File(int run_no);
    void Deallocate();

    double* aircraft_longitude;
    double* aircraft_latitude;
    double* aircraft_altitude;

    double max_straight_speed;
    double max_ascend_speed;
    double start_alt;

    int random_t_1st;

    double random_long;
    double random_lat;
    double random_alt;
    double heading_angle;

    double min_cube_long;
    double max_cube_long;
    double min_cube_lat;
    double max_cube_lat;
    double min_cube_alt;
    double max_cube_alt;

    int depart_or_arrive;

    int* collision_index;

    double* longitude_vector;
    double* latitude_vector;
    double* altitude_vector;
    double* speed_vector;
    double* heading_vector;

    public:

    void SetInitialParameters(string FilePath_input, int Vector_length_input, int TotalCols_input, int drone_index_input, int aircraft_index_input, int takeoff_time_input, double* air_long, double* air_lat, double* air_alt, double aircraft_radius_input, double drone_radius_input);
    void Simulation(int number_runs, double* total_collisions);
    void ClearOutput(int Aircraft_Index);
    void ClearOutput_1File();


};

#endif
