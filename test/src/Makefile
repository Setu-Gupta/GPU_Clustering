SRC=$(wildcard ./*.cpp) $(wildcard ../../src/*.cpp) $(wildcard ./*.cu) $(wildcard ../../src/*.cu)

CC=nvcc

test: $(SRC)
	$(CC) -o test $(SRC) -O3 -Wno-deprecated-declarations -Xcompiler -O3
clean:
	rm test *.clusters
