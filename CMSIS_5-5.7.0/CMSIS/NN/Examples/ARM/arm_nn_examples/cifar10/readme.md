Execute os seguintes comandos **neste diretorio** para obter a classificação das 1000 primeiras imagens do dataset de testes do CIFAR-10.

> GCC é opcional.

```
python3 input_generator.py
```
```
gcc -o teste arm_nnexamples_cifar10.cpp
```
```
gcc -lpthread -o teste arm_nnexamples_cifar10.cpp
```
>    OpenMP
Compilar com GCC
```
gcc -o testecopy arm_nnexamples_cifar10_open_mp.cpp -fopenmp -lstdc++
```
Compilar com clang++

```
clang++ arm_nnexamples_cifar10_open_mp.cpp -fopenmp -o testecopy

```
Executar
```
./teste

```
