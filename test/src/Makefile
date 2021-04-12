SRC=$(wildcard ./*.cpp) $(wildcard ../../src/*.cpp) $(wildcard ./*.cu) $(wildcard ../../src/*.cu)

CC=nvcc

test: $(SRC)
	$(CC) -o test $(SRC) -g
clean:
	rm test *.clusters
