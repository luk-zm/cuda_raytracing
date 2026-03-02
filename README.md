# cuda_raytracing
Ten program implementuje algorytm śledzenia ścieżek (path-tracing) w celu generowania grafiki trójwymiarowej.
# Zależności
- CUDA Toolkit
Testowany z wersją 12.9.
# Budowanie
W głównym katalogu projektu:
Windows:
```
build
```
Domyślnie zbudowany jest zoptymalizowany plik wykonalny.
W celu debugowania należy wywołać:
```
build debug
```

Linux:
```
make
```
albo w celu debugowania:
```
make debug
```

# Używanie
Generowanie domyślnej sceny:
```
./raytracing
```
Z wyborem sceny:
```
./raytracing NRSCENY
```
Z wyborem jakości:
```
./raytracing NRSCENY bLICZBA [sLICZBA]
```
albo
```
./raytracing NRSCENY sLICZBA [bLICZBA]
```
Liczba po 's' określa ilość próbek na piksel, po 'b' głębokość odbić.
By określać swoją jakość trzeba wybrać konkretną scenę.
