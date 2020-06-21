import numpy as np
from keras.utils import to_categorical


def spliting_input(input_aray):


    input_part_one=[]
    input_part_two=[]

    for k in range(len(input_aray)):
    
        swap_axis02=np.swapaxes(input_aray[k][:2],0,2)
        part_one_swap_axis=np.swapaxes(swap_axis02,0,1)
        input_part_one.append(part_one_swap_axis)

        swap_axis02=np.swapaxes(input_aray[k][2:],0,2)
        part_two_swap_axis=np.swapaxes(swap_axis02,0,1)
        input_part_two.append(part_two_swap_axis)

    return input_part_one,input_part_two


def name_it_later(examples_evaluation,labels_evaluation):
    extented_evaluation_set=[]
    extented_label_set=[]

    for i in range(len(examples_evaluation)):
        for k in range(len(examples_evaluation[i])):
            extented_evaluation_set.extend(examples_evaluation[i][k])
            extented_label_set.extend(labels_evaluation[i][k])


    evaluation_extented_first_slice_input,evaluation_extented_second_slice_input=spliting_input(extented_evaluation_set)

    evaluation_encoded_labels = to_categorical(extented_label_set)

    X = [evaluation_extented_first_slice_input, evaluation_extented_second_slice_input]
    Y = evaluation_encoded_labels

    return X,Y


#load dataset


examples_test__0,labels_test_0 = np.load("NPY_files\loaded_Test_0.npy")
examples_test__1,labels_test_1 = np.load("NPY_files\loaded_Test_1.npy")


test_0=name_it_later(examples_test__0, labels_test_0)
np.save("NPY_files\prepared_test0_dataset.npy", test_0)


test_1=name_it_later(examples_test__1, labels_test_1)
np.save("NPY_files\prepared_test1_dataset.npy", test_1)

print("done,its time to evaluate your model")
print("run evaluate model.py,make sure you assign the model folder name to val_acc on evaluate_model.py before run")













