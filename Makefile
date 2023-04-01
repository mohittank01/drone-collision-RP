CC       = g++
CXX      = g++
CXXFLAGS = -g -Wall -O3 -ftree-vectorize
LDLIBS   = 
TARGET   = Monte_Carlo


default: $(TARGET)
all: $(TARGET)

$(TARGET): $(TARGET).o Aircraft.o Drone.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDLIBS)

.PHONY: clean


run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o