gcc -c init.c -o init.o
g++ -c renderer.cpp $(sdl2-config --cflags) -o renderer.o
nvcc -ccbin g++-11 main.cu init.o renderer.o -o nbody $(sdl2-config --libs) -lm
