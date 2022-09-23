#instalar tensorflow
#pip install tensorflow
from keras.datasets import cifar10

def load_dataset():	
	(trainX, trainY), (testX, testY) = cifar10.load_data()	
	return trainX, trainY, testX, testY

trainX, trainY, testX, testY = load_dataset()

counter = [0 for x in range(10)]
images = []
classes = []

for img, classe in zip(trainX, trainY):
    if counter[classe[0]] < 3:
        images.append(img.flatten())
        classes.append(classe[0])
        counter[classe[0]]+=1

    if sum(counter) == 1000:
        break

for img, classe in zip(trainX, trainY):
    if counter[classe[0]] < 100:
        images.append(img.flatten())
        classes.append(classe[0])
        counter[classe[0]]+=1

    if sum(counter) == 1000:
        break

aux = '#define CLASSE {'+str(classes)[1:-1]+'}\n\n'

aux += '#define IMG_DATA {'
for img in images:
    aux += '{'+str(list(img))[1:-1]+'},'

aux = aux[:-1]
aux+='}'

with open('custom_cifar_10_inputs.h', 'w') as f:
    f.write(aux)