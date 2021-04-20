SRC=$(wildcard ./*.cpp) $(wildcard ../../src/*.cpp) $(wildcard ./*.cu) $(wildcard ../../src/*.cu)

CC=nvcc

default: opt

debug: $(SRC)
	$(CC)  -o test $(SRC)  -Wno-deprecated-declarations  -rdc=true -lcudadevrt -g -G

opt: $(SRC)
	$(CC)  -o test $(SRC)  -Wno-deprecated-declarations  -rdc=true -lcudadevrt -O3 -Xcompiler -O3

clean:
	rm test *.clusters
