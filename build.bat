@echo off
setlocal enabledelayedexpansion
cd /D "%~dp0"

set flags=-arch sm_61

for %%a in (%*) do set "%%~a=1"
if "%debug%"=="1" set flags=%flags% -g -G

pushd bin
nvcc %flags% ..\raytracing.cu -DWINDOWS -I ..\include -o "raytracing.exe"
popd
