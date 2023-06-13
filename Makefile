CC       = g++
CXX      = g++
CXXFLAGS = -g -Wall -O3 -ftree-vectorize -fconcepts-ts
LDLIBS   = -lstdc++ -fopenmp -lboost_program_options -lboost_iostreams -lboost_filesystem -lboost_system -lgfortran
TARGET   = Monte_Carlo


default: $(TARGET)
all: $(TARGET)

$(TARGET): $(TARGET).o Aircraft.o Drone.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDLIBS)

.PHONY: clean clean_drone


run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o Drone_coords*.csv *.txt

clean_drone:
	rm Drone_Collisions/Drone_coords*.csv