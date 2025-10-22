all: raytracing.c
	gcc -g -lm raytracing.c -o bin/raytracer

run: all
	cd bin && ./raytracer
