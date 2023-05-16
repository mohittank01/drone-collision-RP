clear
clc
close all

distance_from_airport = "1";

drone_data = importdata("Drone_Collisions_" + distance_from_airport + "_km/Drone_coords_0.csv");
drone_position = importdata("drone_inital_positions_1km.csv");
aircraft_data = importdata("TAKEOFF_UTM_DEPART_A320 LHR time - 2017-06-30 04-00 to 2017-06-30 23-45.csv");

aircraft_index = 2;

callsign = string(aircraft_data.textdata(:,4));
longitude = str2double(string(aircraft_data.textdata(:,15)));
latitude = str2double(string(aircraft_data.textdata(:,14)));
altitude = str2double(string(aircraft_data.textdata(:,8)));

drone_pos_longitude = drone_position.data(:,2);
drone_pos_latitude = drone_position.data(:,3);

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
plot3(longitude(aircraft_index(index_select):aircraft_index(index_select+1)-1), latitude(aircraft_index(index_select):aircraft_index(index_select+1)-1), altitude(aircraft_index(index_select):aircraft_index(index_select+1)-1),'r','DisplayName','Aircraft')
hold on
xlabel("Longitude / UTM")
ylabel("Latitude / UTM")
zlabel("Altitude/ m")
legend
view([35 25 5])
aircraft_dot = plot3(longitude(aircraft_index(index_select)), latitude(aircraft_index(index_select)), altitude(aircraft_index(index_select)),'o','MarkerFaceColor','red','HandleVisibility','off');
grid on
plot3(drone_data(1:1*vector_length,1),drone_data(1:1*vector_length,2),drone_data(1:1*vector_length,3),'-b','DisplayName','Drone')
drone_dot = plot3(drone_data(1,1),drone_data(1,2),drone_data(1,3),'o','MarkerFaceColor','black','HandleVisibility','off');
drone_long = drone_data(1:1*vector_length,1);
drone_lat = drone_data(1:1*vector_length,2);
drone_alt = drone_data(1:1*vector_length,3);
for k = 2:vector_length
    aircraft_dot.XData = longitude(aircraft_index(index_select)+k);
    aircraft_dot.YData = latitude(aircraft_index(index_select)+k);
    aircraft_dot.ZData = altitude(aircraft_index(index_select)+k);
    drone_dot.XData = drone_long(k);
    drone_dot.YData = drone_lat(k);
    drone_dot.ZData = drone_alt(k);
    drawnow 
end

for i = 1:drone_data(end,1)-1
   drone_long = drone_data(i*vector_length+1:(i+1)*vector_length,1);
   drone_lat = drone_data(i*vector_length+1:(i+1)*vector_length,2);
   drone_alt = drone_data(i*vector_length+1:(i+1)*vector_length,3);
   plot3(drone_long,drone_lat,drone_alt,'-b','HandleVisibility','off')
   drone_dot = plot3(drone_data(i*vector_length+1,1),drone_data(i*vector_length+1,2),drone_data(i*vector_length+1,3),'o','MarkerFaceColor','black','HandleVisibility','off');

   for k = 2:vector_length
        aircraft_dot.XData = longitude(aircraft_index(index_select)+k);
        aircraft_dot.YData = latitude(aircraft_index(index_select)+k);
        aircraft_dot.ZData = altitude(aircraft_index(index_select)+k);
        drone_dot.XData = drone_long(k);
        drone_dot.YData = drone_lat(k);
        drone_dot.ZData = drone_alt(k);
        drawnow 
   end
end
hold off

axis manual



figure
plot(longitude(aircraft_index(index_select):aircraft_index(index_select+1)-1), latitude(aircraft_index(index_select):aircraft_index(index_select+1)-1),'r','DisplayName','Aircraft')
hold on
grid on
title("Longitude vs Latitude")
plot(drone_data(1:1*vector_length,1),drone_data(1:1*vector_length,2),'-b','DisplayName','Drone') 
for i = 1:drone_data(end,1)-1
   plot(drone_data(i*vector_length+1:(i+1)*vector_length,1),drone_data(i*vector_length+1:(i+1)*vector_length,2),'-b','HandleVisibility','off')
   plot(drone_data(i*vector_length+1,1),drone_data(i*vector_length+1,2),'rx','HandleVisibility','off') 
end
% for i = 1:length(drone_pos_latitude)
%    plot(drone_pos_longitude(i),drone_pos_latitude(i),'rx','HandleVisibility','off') 
% end
axis equal
legend
hold off

figure
plot(longitude(aircraft_index(index_select):aircraft_index(index_select+1)-1), altitude(aircraft_index(index_select):aircraft_index(index_select+1)-1),'r','DisplayName','Aircraft')
hold on
grid on
title("Longitude vs Altitude")
plot(drone_data(1:1*vector_length,1),drone_data(1:1*vector_length,3),'-b','DisplayName','Drone') 
for i = 1:drone_data(end,1)-1
   plot(drone_data(i*vector_length+1:(i+1)*vector_length,1),drone_data(i*vector_length+1:(i+1)*vector_length,3),'-b','HandleVisibility','off')
   plot(drone_data(i*vector_length+1,1),drone_data(i*vector_length+1,3),'rx','HandleVisibility','off') 
end
legend
hold off

figure
plot(latitude(aircraft_index(index_select):aircraft_index(index_select+1)-1), altitude(aircraft_index(index_select):aircraft_index(index_select+1)-1),'r','DisplayName','Aircraft')
hold on
grid on
title("Latitude vs Altitude")
plot(drone_data(1:1*vector_length,2),drone_data(1:1*vector_length,3),'-b','DisplayName','Drone') 
for i = 1:drone_data(end,1)-1
   plot(drone_data(i*vector_length+1:(i+1)*vector_length,2),drone_data(i*vector_length+1:(i+1)*vector_length,3),'-b','HandleVisibility','off')
   plot(drone_data(i*vector_length+1,2),drone_data(i*vector_length+1,3),'rx','HandleVisibility','off') 
end
legend
hold off