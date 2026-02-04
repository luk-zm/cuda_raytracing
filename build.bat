@echo off

pushd bin
nvcc -g -G -arch sm_61 ..\raytracing.cu -DWINDOWS -I ..\include -o "raytracing.exe"
REM nvcc -arch sm_61 ..\raytracing.cu -DWINDOWS -I ..\include -o "raytracing.exe"
popd
