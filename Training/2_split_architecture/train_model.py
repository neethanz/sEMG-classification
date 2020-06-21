import os
import numpy as np
import keras 
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, BatchNormalization, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.models import model_from_json
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size=300
epochs=20
no_of_filters_in_split=120
no_of_filters_concat=240
drop_out=0.5

print("")
print("Start training...")
print("")

[X,Y,X_validate,Y_validate] = np.load('../NPY_files/prepared_training_dataset.npy')

first_slice_input = Input(shape=(8,14, 2))  ## branch 1 with image input
l1 = Conv2D(120, (4, 3))(first_slice_input)
l1 = BatchNormalization()(l1)
l1 = PReLU()(l1)
l1 = Dropout(drop_out)(l1)

l1 = Conv2D(120, (3, 3))(l1)
l1 = BatchNormalization()(l1)
l1 = PReLU()(l1)
l1 = Dropout(drop_out)(l1)


second_slice_input = Input(shape=(8,14, 2))## branch 1 with image input
l2 = Conv2D(120, (4, 3))(second_slice_input)
l2 = BatchNormalization()(l2)
l2 = PReLU()(l2)
l2 = Dropout(drop_out)(l2)

l2 = Conv2D(120, (3, 3))(l2)
l2 = BatchNormalization()(l2)
l2 = PReLU()(l2)
l2 = Dropout(drop_out)(l2)


concatenated = concatenate([l1,l2])

x = Conv2D(240, (3, 3))(concatenated)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Dropout(drop_out)(x)

x = Flatten()(x)
x = Dense(100)(x)
x = Dense(100)(x)
output_layer = Dense(7, activation='softmax')(x)

model = Model([first_slice_input, second_slice_input],output_layer)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


results=model.fit(X, Y, batch_size = batch_size, epochs =epochs,validation_data = (X_validate, Y_validate))


path = str(np.round_(results.history['val_acc'][-1],decimals = 4))

print("")

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


data = {}
data['accuracy_metrics']=results.history
data['details']={

	'no_of_filters_in_split': no_of_filters_in_split,
    'no_of_filters_concat': no_of_filters_concat,
    'drop_out': drop_out,
    'batch_size':batch_size,
    'epochs': epochs
}

with open(str(path)+'/data.txt', 'w') as outfile:
    json.dump(data, outfile)


model_json = model.to_json()	
with open(str(path)+"/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(str(path)+"/model.h5")
print("Saved model to disk")


print("Done training...")
print("")