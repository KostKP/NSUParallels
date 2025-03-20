# Компилятор
CXX = g++
# Флаги компиляции
CXXFLAGS = -std=c++20 -Wall -O2

# Правило по умолчанию
all:
	$(CXX) $(CXXFLAGS) -o TASK task.cpp

test:
	$(CXX) $(CXXFLAGS) -o TEST test.cpp
	-./TEST
	rm -f TEST

clean:
	rm -f TASK
