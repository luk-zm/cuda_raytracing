@echo off
setlocal enabledelayedexpansion
cd /D "%~dp0"

set flags=-arch native

for %%a in (%*) do set "%%~a=1"
if "%debug%"=="1" set flags=%flags% -g -G

if not exist bin mkdir bin

pushd bin
nvcc %flags% ..\src\main.cu -I ..\include -o "raytracing.exe"
popd
