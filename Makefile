all: raytracing.c
	gcc -g -lm raytracing.c -o bin/raytracer

rel: raytracing.c
	gcc -lm raytracing.c -O3 -o bin/raytracer

run: all
	cd bin && ./raytracer
