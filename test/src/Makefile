SRC=$(wildcard ./*.cpp) $(wildcard ../../src/*.cpp)

CC=g++

test: $(SRC)
	$(CC) -o test $(SRC) -g
clean:
	rm test *.clusters
