CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -I. -Iexternal

TARGET = test_frame
TEST = test_frame.cpp

all: $(TARGET)

$(TARGET): $(TEST)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)