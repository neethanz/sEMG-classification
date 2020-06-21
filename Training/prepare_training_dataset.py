import numpy as np
from keras.utils import to_categorical

def scramble(examples, labels):  
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])
    return new_examples, new_labels


def split_train_valid(examples, labels):
  
    X_training, Y_gesture = [], []
    X_validation, Y_gesture_validation = [], []

    for j in range(40):

        examples_personne_training = []
        labels_gesture_personne_training = []

        examples_personne_valid = []
        labels_gesture_personne_valid = []
		
        for k in range(len(examples[j])):
 
            if k < 21:
                examples_personne_training.extend(examples[j][k]) 
                labels_gesture_personne_training.extend(labels[j][k])
            else:
                examples_personne_valid.extend(examples[j][k])
                labels_gesture_personne_valid.extend(labels[j][k])

        examples_personne_scrambled, labels_gesture_personne_scrambled = scramble(
             examples_personne_training, labels_gesture_personne_training)

        examples_personne_scrambled_valid, labels_gesture_personne_scrambled_valid = scramble(
            examples_personne_valid, labels_gesture_personne_valid)


        X_training.append(examples_personne_scrambled)
        Y_gesture.append(labels_gesture_personne_scrambled)

        X_validation.append(examples_personne_scrambled_valid)
        Y_gesture_validation.append(labels_gesture_personne_scrambled_valid)

    datasets = [(X_training, Y_gesture),
                (X_validation, Y_gesture_validation)]
    return datasets


def spliting_input(input_aray):


    input_part_one=[]
    input_part_two=[]

    for k in range(len(input_aray)): 

        for j in range(len(input_aray[k])):

            swap_axis02=np.swapaxes(input_aray[k][j][:2],0,2)
            part_one_swap_axis=np.swapaxes(swap_axis02,0,1)
            swap_axis02=np.swapaxes(input_aray[k][j][2:],0,2)
            part_two_swap_axis=np.swapaxes(swap_axis02,0,1)

            input_part_one.append(part_one_swap_axis)
            input_part_two.append(part_two_swap_axis)

    return input_part_one,input_part_two


#load dataset

print("")
print("preparing recipe...")
print("")

datasets_pre_training = np.load("NPY_files\loaded_training_dataset.npy")
examples_pre_training, labels_pre_training = datasets_pre_training



[(X_training, Y_gesture),
                (X_validation, Y_gesture_validation)]=split_train_valid(examples_pre_training, labels_pre_training)


input_part_one,input_part_two=spliting_input(X_training)
validate_part_one,validate_part_two=spliting_input(X_validation)

train_extend_labels=[]
validate_extend_labels=[]

for i in range(len(Y_gesture)):

    train_extend_labels.extend(Y_gesture[i])
    validate_extend_labels.extend(Y_gesture_validation[i])
  

train_encoded_labels = to_categorical(train_extend_labels)
validate_encoded_labels = to_categorical(validate_extend_labels)


X=[input_part_one, input_part_two]
Y = train_encoded_labels
X_validate=[validate_part_one,validate_part_two]
Y_validate=validate_encoded_labels


#prepared data set for training

prepared_dataset=[X,Y,X_validate,Y_validate]

np.save("NPY_files\prepared_training_dataset.npy", prepared_dataset)

print("Yap done,its Time for Train....")
print('Now run train_model.py at 2_split_architecture')
print("")

