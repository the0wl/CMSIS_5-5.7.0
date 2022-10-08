#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

void* thread(void* arg);

int main(int argc, char *argv[])
{
  printf("start execution\n");

  pthread_t* ptid;
  ptid = (pthread_t*) malloc(1000 * sizeof(pthread_t));

  int j;

  for (j=0; j < 1000; j++) { // KELVIN

    pthread_create(&ptid[j], NULL, thread, &j);  

  }

  for (j=0; j < 1000; j++) { // KELVIN

    pthread_join(ptid[j], NULL);

  }

  free(ptid);

  return 0;
}

void* thread(void* arg) {
    int j = *((int *) arg);
    printf("Imagem %d\n", j);
    
    pthread_exit(NULL);
}