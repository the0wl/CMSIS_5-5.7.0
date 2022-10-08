#instalar tensorflow
#pip install tensorflow
from keras.datasets import cifar10

def load_dataset():	
	(trainX, trainY), (testX, testY) = cifar10.load_data()	
	return trainX, trainY, testX, testY

trainX, trainY, testX, testY = load_dataset()

counter = 0
images = []
classes = []

for img in trainX:
    counter += 1
    images.append(img.flatten())

    if counter == 1000:
        break

aux = '#define IMG_DATA {'
for img in images:
    aux += '{'+str(list(img))[1:-1]+'},'

aux = aux[:-1]
aux+='}'

with open('custom_cifar_10_inputs.h', 'w') as f:
    f.write(aux)