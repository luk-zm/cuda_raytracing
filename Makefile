rel: src/main.cu | bin_folder
	nvcc -arch native -lm src/main.cu -O3 -I include -o bin/raytracing

debug: src/main.cu | bin_folder
	nvcc -arch native -g -G -lm src/main.cu -I include -o bin/raytracing

run: rel
	cd bin && ./raytracing

bin_folder:
	mkdir -p bin
