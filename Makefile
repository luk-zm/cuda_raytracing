rel: src/raytracing.cu | bin_folder
	nvcc -arch native -lm raytracing.c -O3 -o bin/raytracing

debug: src/raytracing.cu | bin_folder
	nvcc -arch native -g -lm raytracing.c -o bin/raytracing

run: rel
	cd bin && ./raytracing

bin_folder:
	mkdir -p bin
