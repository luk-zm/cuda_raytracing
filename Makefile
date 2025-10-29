all: raytracing.c
	gcc -DLINUX -g -lm raytracing.c -o bin/raytracer

rel: raytracing.c
	gcc -DLINUX -lm raytracing.c -O3 -o bin/raytracer

run: all
	cd bin && ./raytracer
