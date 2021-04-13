SRC=$(wildcard ./*.cpp) $(wildcard ../../src/*.cpp) $(wildcard ./*.cu) $(wildcard ../../src/*.cu)

CC=nvcc

test: $(SRC)
	$(CC) -g -G  -o test $(SRC)  -Wno-deprecated-declarations  -rdc=true -lcudadevrt
clean:
	rm test *.clusters
