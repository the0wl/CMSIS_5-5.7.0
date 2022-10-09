# Comparação de desempenho em implementações multi-threads no algoritmo `arm_nnexamples_cifar10`

O presente repositório tem como finalidade comparar a utilização das bibliotecas openmp e pthread para realizar a paralelização de um algoritmo de rede neural convolucional (CNN) que trabalha sobre o coleção de imagens do Instituto Canadense para Pesquisa Avancada (CIFAR), chamada CIFAR-10. Foram desenvolvidos dois algoritmos, um com cada biblioteca. Após isso, os tempos de execução foram comparados, utilizando 1, 2, 4, 8 threads. Por fim, foram apresentados e discutidos os resultados obtidos.

O repositório original se encontra em [CMSIS](https://github.com/ARM-software/CMSIS_5/tree/5.7.0).

## Introdução

O presente trabalho tem como objetivo comparar a utilização e aplicação de paralelismo em um algoritmo de machine learning e visão computacional, tendo como base a biblioteca Common Microcontroller Software Interface Standard (CMSIS). O CMSIS é um conjunto de ferramentas, estruturas, e fluxos de trabalho que ajudam a simplificar a reutilização de software, reduzir a curva de aprendizagem para programadores de microcontroladores, acelerarando a construção e depuração de projetos.

Para a comparacao foi tomado como base o algoritmo arm_nnexamples_cifar10, presente nos exemplos de implementação de redes neurais, em arquiteturas Advanced RISC Machine (ARM). A partir do código base, que implementa o reconhecimento de uma imagem, fixa na biblioteca, foi implementada uma versão capaz de reconhecer 1000 imagens. Estas imagens são as 1000 primeiras imagens da coleção de imagens CIFAR-10.

Esta nova versão foi paralelizada utilizando as bibliotecas omp.h e pthread.h do C++. Também foram utilizados os compiladores gcc e clang, comparando a execucao em 1, 2, 4 e 8 threads.

## Ambiente de desenvolvimento

Como sistema operacional, foi utilizado o `Windows 10`, rodando o recurso Windows Subsystem for Linux (WSL) com a distribuição `Ubuntu 22.04`. Para configurar o seu sistema de acordo com o utilizado neste trabalho você deve:
- Instalar o recurso [Windows Subsystem for Linux (WSL)](#instalação-do-windows-subsystem-for-linux-wsl).
- Instalar o editor [Visual Studio Code](https://code.visualstudio.com) (VSCode).
- A instalação da extensão [WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) no VSCode.
- Dentro da paleta de comandos procure por `WSL: New WSL window using distro` e escolha `Ubuntu 22.04`.
- Navegue até a pasta `CMSIS_5-5.7.0/CMSIS_5-5.7.0/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/`.

## Instalação do Windows Subsystem for Linux (WSL)

No PowerShell em modo administrador execute o comando abaixo para realizar o download e a instalação:

```shell
wsl –install
```

> Pode ser necessária a reinicialização do computador.

Na barra de pesquisa, pesquise por `WSL`. Será aberto um terminal onde será solicitado uma senha para o usuário padrão da instalação da distro utilizada.

## Preparação dos dados de entrada

Foi criado o arquivo `input_generator.py` para realizar a leitura das imagens do dataset `cifar10` disponível na biblioteca `keras.datasets` do python. 

O comando a seguir irá fazer com que as primeiras 1000 imagens do dataset sejam processadas e transformadas em um array de imagens (no formato de pixels) para o treinamento na rede neural construída em `c++`. 

```shell
python3 input_generator.py
```

## Compilação os algoritmos

Foram analisadas duas implementações neste trabalho. Uma utiliza a biblioteca `omp` (openmp) e a outra a biblioteca `pthread`.

Abaixo estão dispostos os comandos de terminal em gcc ou clang para compilar o algoritmo utilizando a biblioteca `omp`.

```shell
gcc -o algoritmo arm_nnexamples_cifar10_open_mp.cpp -fopenmp -lstdc++
```

```shell
clang++ arm_nnexamples_cifar10_open_mp.cpp -fopenmp -o algoritmo
```

Para compilar o algoritmo que utiliza a biblioteca `pthread` utilize os respectivos comandos de terminal em gcc ou clang.

```shell
gcc -pthread -o algoritmo arm_nnexamples_cifar10_pthread.cpp
```

```shell
clang++ -pthread -o algoritmo arm_nnexamples_cifar10_pthread.cpp
```

## Execução dos algoritmos

Qualquer um dos comandos anteriores irá gerar um arquivo de output chamado `algoritmo`. Para executar o arquivo utilize o comando de terminal a seguir.

```shell
./algoritmo
```

O algoritmo irá solicitar o número de threads a serem utilizadas. Basta informar um número de threads a serem utilizadas e apertar a tecla `ENTER`.

> O número de threads é um número positivo e que será limitado de acordo com as especificações do processador da máquina onde o arquivo está sendo executado.

## Análise dos resultados

Os resultados obtidos foram dispostos na tabela abaixo.

| Biblioteca (compilador)  | 1 thread | 2 threads | 3 threads | 4 threads | 8 threads |
|:------------------------:|:--------:|:---------:|:---------:|:---------:|:---------:|
| omp (gcc)                |   24.36  |   14.78   |   10.78   |    8.77   |    6.74   |
| omp (clang)              |   25.87  |   13.57   |   10.24   |    8.71   |    6.78   |
| pthread (gcc)            |   24.63  |   13.4    |   10.96   |    9.35   |    6.55   |
| pthread (clang)          |   23.86  |   13.58   |   10.90   |    9.17   |    6.39   |

Como resultado das execuções e análises realizadas das bibliotecas OpenMp e pthread, executando com diferentes números de threads, podemos ver uma pequena diferença de tempo de execução entre as bibliotecas utilizadas. Mas comparando a execução em single thread com a execução paralela podemos ver uma redução significativa no tempo de execução, mostrando um bom ganho de eficiência e desempenho.

O melhor resultado obtido foi utilizando a quantidade máxima de threads que a máquina de teste permitia, neste caso, 8 threads. O resultado deste teste com máxima eficiência foi o tempo de execução de aproximadamente 6 segundos, com pequenas variações entre os compiladores e bibliotecas utilizadas.

Diante destes resultados, o grupo constatou que no caso de estudo, a execução por meio da biblioteca `omp` se destaca em comparação a biblioteca `pthread`. Podemos observar resultados muito semelhantes, exibindo diferenças de milisegundos, porém a implementação de multi-threads utilizando a biblioteca `omp` foi muito mais simples, necessitando apenas uma linha de código e sem se preocupar com o `lock` de recursos, como é o caso da biblioteca `pthread`.
