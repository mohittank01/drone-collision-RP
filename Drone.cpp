#include "Drone.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>

using namespace std;

void Drone::SetInitialConditions(string FilePath_input, int Vector_length_input, int TotalCols_input, int drone_index_input, int takeoff_time_input, double* air_long, double* air_lat, double* air_alt){
    FilePath = FilePath_input;
    VectorLength = Vector_length_input;
    TotalCols = TotalCols_input;
    DroneIndex = drone_index_input;
    TakeoffTime = takeoff_time_input;

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

    aircraft_longitude = new double[VectorLength];
    aircraft_latitude = new double[VectorLength];
    aircraft_altitude = new double[VectorLength];

    for(int i = 0; i < VectorLength; ++i){
        aircraft_longitude[i] = air_long[i];
        aircraft_latitude[i] = air_lat[i];
        aircraft_altitude[i] = air_alt[i];
    }

    if(aircraft_altitude[0] != 0){
        depart_or_arrive = 1; // ARRIVAL 
    }
    else{
        depart_or_arrive = 0; // DEPART
    }
 
    longitude_factor_1st = new int[1];
    latitude_factor_1st = new int[1];

    start_alt = 100.0; // Starting Altitude
    max_straight_speed = 19.0;

    // Initialise vectors
    longitude_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + longitude_col_no]); // Initialising longitude 
    latitude_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + latitude_col_no]); // Initialising latitude
    heading_vector[0] = stod(PositionData[((DroneIndex+1)*TotalCols) + heading_col_no]); // Initialising heading
    altitude_vector[0] = start_alt; // 100m starting altitude
    speed_vector[0] = max_straight_speed; // 19m/s max strightline speed


    max_ascend_speed = 8.0;

    FirstStage();
    SecondStage();
    Output();
    Deallocate();
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

// DONT ACTUIALLY NEED THIS FUCNTION BUT KEPT IT JUST IN CASE DOUBT ID NEED IT
void Drone::HeadingCalc(double heading_val, int* longitude_factor, int* latitude_factor){
    // Top right quadrant
    if(heading_val >= M_PI && heading_val <= 1.5*M_PI){
        *longitude_factor = -1;
        *latitude_factor = -1;
    }
    // Bottom right quadrant
    else if(heading_val >= 1.5*M_PI && heading_val <= 2*M_PI){
        *longitude_factor = -1;
        *latitude_factor = 1;
    }
    // Bottom left quadrant
    else if(heading_val >= 0 && heading_val <= 0.5*M_PI){
        *longitude_factor = 1;
        *latitude_factor = 1;
    }
    // Top left quadrant
    else if(heading_val >= 0.5*M_PI && heading_val <= M_PI){
        *longitude_factor = 1;
        *latitude_factor = -1;
    }
}


void Drone::FirstStage(){
    int min_t_1st = 0;
    int max_t_1st = TakeoffTime;

    random_device seed;
    mt19937 engine(seed());
    uniform_int_distribution<int> dist(min_t_1st, max_t_1st); // uniform, unbiased

    random_t_1st = dist(engine);

    for(int i = 1; i < random_t_1st; ++i){
        longitude_vector[i] = longitude_vector[i-1] + sin(heading_vector[i-1])*max_straight_speed;
        latitude_vector[i] = latitude_vector[i-1] + cos(heading_vector[i-1])*max_straight_speed;
        heading_vector[i] = heading_vector[i-1];
        speed_vector[i] = max_straight_speed;
        altitude_vector[i] = start_alt;
    }
}

void Drone::CubedVolume(){
    // LHR
    min_cube_alt = 100;
    max_cube_alt = min_cube_alt + 400; // ADDING 400m TO THE MIN ALTITUDE

    // 27R
    if(aircraft_latitude[0] <= 5706340.62 && aircraft_latitude[0] >= 5705792.58){
        min_cube_lat = 5705770.46;
        max_cube_lat = min_cube_lat + 500; // ADDING 500m TO THE MIN FOR THE RUNWAY WIDTH
        if(depart_or_arrive){ // ARRIVAL
            min_cube_long = 676345.60;
            max_cube_long = min_cube_long + 5000;
        }
        else{ // DEPARTURE
            max_cube_long = 676819.02; 
            min_cube_long = max_cube_long - 5000; // FOR THE LENGTH OF THE CUBED VOLUME - 5000m
        }
    }
    // 27L
    else if(aircraft_latitude[0] <= 5705065.74 && aircraft_latitude[0] >= 5704388.38){
        max_cube_lat = 5704800.46;
        min_cube_lat = max_cube_lat - 500; // MINUS 500m TO THE MAX FOR THE RUNWAY WIDTH

        if(depart_or_arrive == 1){ // ARRIVAL
            min_cube_long = 676345.60;
            max_cube_long = min_cube_long + 5000;
        }
        else{ // DEPARTURE
            max_cube_long = 676819.02; 
            min_cube_long = max_cube_long - 5000; // FOR THE LENGTH OF THE CUBED VOLUME - 5000m
        }
    }
}


void Drone::SecondStage(){
    CubedVolume();

    // Seed generator
    random_device seed;
    mt19937 engine(seed());
    
    // Uniform distribution - LONGITUDE
    uniform_real_distribution<double> long_dist(min_cube_long, max_cube_long); // uniform, unbiased

    // Uniform distribution - LATITUDE
    uniform_real_distribution<double> lat_dist(min_cube_lat, max_cube_lat); // uniform, unbiased

    // Uniform distribution - ALTITUDE
    uniform_real_distribution<double> alt_dist(min_cube_alt, max_cube_alt); // uniform, unbiased

    random_long = long_dist(engine);
    random_lat = lat_dist(engine);
    random_alt = alt_dist(engine);

    cout.precision(15);
    cout << random_long << endl;
    cout << random_lat << endl;
    cout << random_alt << endl;

    double pitch_angle;
    double modulus_long_lat;

    int last_index = random_t_1st - 1;

    double long_diff = abs(random_long - longitude_vector[last_index]);
    double lat_diff = abs(random_lat - latitude_vector[last_index]);
    double alt_diff = abs(random_alt - altitude_vector[last_index]);


    cout<<longitude_vector[last_index]<<endl;
    cout<<latitude_vector[last_index]<<endl;

    // TOP LEFT
    if(random_long <= longitude_vector[last_index] && random_lat >= latitude_vector[last_index]){
        heading_angle = 2*M_PI - atan(long_diff / lat_diff);
        cout << 1 << endl;
    }
    // BOTTOM LEFT
    else if(random_long <= longitude_vector[last_index] && random_lat <= latitude_vector[last_index]){
        heading_angle = 1.5*M_PI - atan(lat_diff / long_diff);
        cout << 2 << endl;
    }
    // BOTTOM RIGHT
    else if(random_long >= longitude_vector[last_index] && random_lat <= latitude_vector[last_index]){
        heading_angle = M_PI - atan(long_diff / lat_diff);
        cout << 3 << endl;
    }
    // TOP RIGHT
    else if(random_long >= longitude_vector[last_index] && random_lat >= latitude_vector[last_index]){
        heading_angle = atan(long_diff / lat_diff);
        cout << 4 << endl;
    }
    
    modulus_long_lat = sqrt(long_diff*long_diff + lat_diff*lat_diff);

    pitch_angle = atan(alt_diff / modulus_long_lat);

    double velocity_factor = (max_straight_speed - (2*(max_straight_speed - max_ascend_speed)/M_PI)*pitch_angle);

    for(int i = random_t_1st; i < VectorLength; ++i){
        longitude_vector[i] = longitude_vector[i-1] + sin(heading_angle) * velocity_factor;
        latitude_vector[i] = latitude_vector[i-1] + cos(heading_angle) * velocity_factor;
        altitude_vector[i] = altitude_vector[i-1] + sin(pitch_angle) * velocity_factor;
    }

    cout << altitude_vector[VectorLength-1] << endl;
    cout << velocity_factor << endl;

}

void Drone::Output(){
    ofstream outfile;
    outfile.open("Drone_coords.txt", ofstream::out | ofstream::trunc);
    for (int i=0; i < VectorLength; ++i){
        outfile << longitude_vector[i] << " " << latitude_vector[i] << " " << altitude_vector[i] << '\n';
    }
    outfile << '\n';
    outfile << random_long << " " << random_lat << " " << random_alt;
    outfile.close();
}

void Drone::Deallocate(){
    delete[] longitude_vector;
    delete[] latitude_vector;
    delete[] altitude_vector;
    delete[] speed_vector;
    delete[] heading_vector;

    delete[] aircraft_longitude;
    delete[] aircraft_latitude;
    delete[] aircraft_altitude;
	
	delete[] longitude_factor_1st;
    delete[] latitude_factor_1st;
}