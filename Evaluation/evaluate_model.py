import numpy as np
import numpy as np
import keras 
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, BatchNormalization, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json

print("")
print("make sure you assinged model folder name to val_acc")
print("")
#set your model folder here
val_acc=0.9435

X0,Y0= np.load("NPY_files\prepared_test0_dataset.npy")
X1,Y1= np.load("NPY_files\prepared_test1_dataset.npy")


json_file = open('../Training/2_split_architecture/'+str(val_acc)+'/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('../Training/2_split_architecture/'+str(val_acc)+'/model.h5')
print("Loaded model from disk")


batch_size=300

with open('../Training/2_split_architecture/'+str(val_acc)+'/data.txt') as json_file:
    data = json.load(json_file)


batch_size=data['details']['batch_size']

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Evaluate the model on the test data using `evaluate`
print("")
print('\n# Evaluate on test data')
print("")
results_0 = loaded_model.evaluate(X0, Y0, batch_size=batch_size)
results_1 = loaded_model.evaluate(X1, Y1, batch_size=batch_size)

print("")
print('test loss_0, test acc_0:', results_0)
print('test loss_1, test acc_1:', results_1)


data['test_results']={
    'test_0_loss,test_0_accuracy':results_0,
    'test_1_loss,test_1_accuracy':results_1
}



with open('../Training/2_split_architecture/'+str(val_acc)+'/data.txt', 'w') as outfile:
    json.dump(data, outfile)

print("")
print("You can find whole details at data.txt on your model folder")
print("")



