@echo off

set FLAGS=-Od -Ob1 -FC -Z7 -nologo 
REM -g -std=c99 -Wall -Wextra -pedantic
REM -lmingw32
set LIBS=shell32.lib

pushd bin
cl ..\raytracing.c %FLAGS% -Fe"raytracing.exe" -I ..\include %LIBS% /link /SUBSYSTEM:CONSOLE 
popd
