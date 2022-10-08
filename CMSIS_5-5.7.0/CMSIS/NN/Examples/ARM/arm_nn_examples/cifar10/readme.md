# Multi-threads no algoritmo cifar10

Execute os seguintes comandos **neste diretorio** para obter a classificação das 1000 primeiras imagens do dataset de testes do CIFAR-10. Estas serão utilizadas como entrada no algoritmo.

```
python3 input_generator.py
```

## Compilar com OpenMP
### Utilizando GCC
```
gcc -o algoritmo arm_nnexamples_cifar10_open_mp.cpp -fopenmp -lstdc++
```
### Utilizando Clang++
```
clang++ arm_nnexamples_cifar10_open_mp.cpp -fopenmp -o algoritmo
```

## Compilar com PThread
### Utilizando GCC
```
gcc -pthread -o algoritmo arm_nnexamples_cifar10_pthread.cpp
```
### Utilizando Clang++
```
clang++ -pthread -o algoritmo arm_nnexamples_cifar10_pthread.cpp
```

## Executar
```
./algoritmo
```

## Resultados
> Biblioteca e Compilador por Tempo de execução em segundos por thread
>> 8 é o número de threads do dispositivo utilizado

|                          |                    1                     |   2   |   3   |  4   |  8   |
|--------------------------|:----------------------------------------:|:-----:|:-----:|:----:|:----:|
|      OpenMP com GCC      |                   24.36                  | 14.78 | 10.78 | 8.77 | 6.74 |
|     OpenMP com Clang     |                   25.87                  | 13.57 | 10.24 | 8.71 | 6.78 |
|      PThread com GCC     |                   24.63                  |  13.4 | 10.96 | 9.35 | 6.55 |
|     PThread com Clang    |                   23.86                  | 13.58 | 10.90 | 9.17 | 6.39 |