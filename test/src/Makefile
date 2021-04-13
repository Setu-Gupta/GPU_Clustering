SRC=$(wildcard ./*.cpp) $(wildcard ../../src/*.cpp) $(wildcard ./*.cu) $(wildcard ../../src/*.cu)

CC=nvcc

test: $(SRC)
	$(CC) -o test $(SRC) -O3 -Wno-deprecated-declarations -Xcompiler -O3 -rdc=true -lcudadevrt
clean:
	rm test *.clusters
