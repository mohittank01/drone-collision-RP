clear
clc

drone_data = importdata("Drone_coords_0.csv");
aircraft_data = importdata("TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv");

aircraft_index = [2];

callsign = string(aircraft_data.textdata(:,4));
longitude = str2double(string(aircraft_data.textdata(:,15)));
latitude = str2double(string(aircraft_data.textdata(:,14)));
altitude = str2double(string(aircraft_data.textdata(:,8)));

for i = 2:length(callsign)+1
    if i == length(callsign)
        aircraft_index(end+1) = i+1;
        break
    end
    if callsign(i) ~= callsign(i+1)
        aircraft_index(end+1) = i+1;
    end
    
end

index_select = 1;
vector_length = length(longitude(aircraft_index(index_select):aircraft_index(index_select+1)-1));


figure
plot3(longitude(aircraft_index(index_select):aircraft_index(index_select+1)-1), latitude(aircraft_index(index_select):aircraft_index(index_select+1)-1), altitude(aircraft_index(index_select):aircraft_index(index_select+1)-1),'r')
hold on
grid on
plot3(drone_data(1:1*vector_length,1),drone_data(1:1*vector_length+1,2),drone_data(1:1*vector_length+1,3),'-b') 
%for i = 1:9999
%   plot3(drone_data(i*vector_length+1:(i+1)*vector_length,1),drone_data(i*vector_length+1:(i+1)*vector_length,2),drone_data(i*vector_length+1:(i+1)*vector_length,3),'-b') 
%end
