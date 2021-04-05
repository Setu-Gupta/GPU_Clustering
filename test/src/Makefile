SRC=$(wildcard ./*.cpp) $(wildcard ../../src/*.cpp)

CC=g++

test: $(SRC)
	$(CC) -o test $(SRC) -O3
clean:
	rm test *.clusters