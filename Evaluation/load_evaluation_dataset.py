import numpy as np 
import sys
number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5

from scipy import signal


def shift_electrodes(examples, labels):

    index_normal_class = [1, 2, 6, 2] 
    class_mean = []

    for classe in range(3,7):    
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.extend(examples[k])
            Y_example.extend(labels[k]) 

        spectrogram_add = []
        for j in range(len(X_example)):
            if Y_example[j] == classe:
                if spectrogram_add == []:
                    spectrogram_add = np.array(X_example[j][0])
                else:
                    spectrogram_add += np.array(X_example[j][0])

        class_mean.append(np.argmax(np.sum(np.array(spectrogram_add), axis=0)))

    new_spectrogram_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10)
    new_spectrogram_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10)

    shifts_array = []
    for valueA, valueB in zip(new_spectrogram_emplacement_left, new_spectrogram_emplacement_right):
        if valueA < valueB:
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            orientation = 1
            shifts_array.append(orientation*valueB)
    final_shifting = np.mean(np.array(shifts_array))
    
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    X_example = []
    Y_example = []
    for k in range(len(examples)): 
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting))
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example

   

def calculate_spectrogram_dataset(dataset):
    dataset_spectrogram = []
    for examples in dataset:
        canals = []
      

        for electrode_vector in examples:

            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                calculate_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)

            spectrogram_of_vector = spectrogram_of_vector[1:]
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))

        example_to_classify = np.swapaxes(canals, 0, 1)
        
        dataset_spectrogram.append(example_to_classify)
    return dataset_spectrogram 


def calculate_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window="hann",
                                                                                       scaling="spectrum")
    return spectrogram_of_vector, time_segment_sample, frequencies_samples



def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    example = []
    emg_vector = []
    for value in vector_to_format:
        emg_vector.append(value)
        if (len(emg_vector) >= 8):
            if (example == []):
                example = emg_vector
            else:
                example = np.row_stack((example, emg_vector))
            emg_vector = []
            if (len(example) >= number_of_vector_per_example):
                
                example = example.transpose()
                dataset_example_formatted.append(example)
                example = example.transpose()
                example = example[size_non_overlap:]
    test = calculate_spectrogram_dataset(dataset_example_formatted)
    return np.array(test) 

print("this process will take nearly 6 min on i7 7700HQ ")
print("")
print("Reading Data from Test_0")
print("")

list_dataset = []
list_labels = []


for candidate in range(16):
    labels = []
    examples = []
    for i in range(number_of_classes * 4):
        data_read_from_file = np.fromfile('..\Dataset\EvaluationDataset\Male'+str(candidate)+'\\Test0\\classe_%d.dat' % i, dtype=np.int16)
        data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
        dataset_example = format_data_to_train(data_read_from_file) 
        examples.append(dataset_example) #28*190*4*8*14
        labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))

    examples, labels = shift_electrodes(examples, labels)
    list_dataset.append(examples)
    list_labels.append(labels)

for candidate in range(2):
    labels = []
    examples = []
    for i in range(number_of_classes * 4):
        data_read_from_file = np.fromfile('..\Dataset\EvaluationDataset\Female'+str(candidate)+'\\Test0\\classe_%d.dat' % i, dtype=np.int16)
        data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
        dataset_example = format_data_to_train(data_read_from_file) 
        examples.append(dataset_example) 
        labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))

    examples, labels = shift_electrodes(examples, labels)
    list_dataset.append(examples)
    list_labels.append(labels)

print("Finished Reading Data from Test_0")
print("")
datasets = [list_dataset, list_labels]
np.save("NPY_files\loaded_Test_0.npy", datasets)




print("Reading Data from Test_1")
print("")

list_dataset = []
list_labels = []


for candidate in range(16):
    labels = []
    examples = []
    for i in range(number_of_classes * 4):
        data_read_from_file = np.fromfile('..\Dataset\EvaluationDataset\Male'+str(candidate)+'\\Test1\\classe_%d.dat' % i, dtype=np.int16)
        data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
        dataset_example = format_data_to_train(data_read_from_file) 
        examples.append(dataset_example) 
        labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))

    examples, labels = shift_electrodes(examples, labels)
    list_dataset.append(examples)
    list_labels.append(labels)

for candidate in range(2):
    labels = []
    examples = []
    for i in range(number_of_classes * 4):
        data_read_from_file = np.fromfile('..\Dataset\EvaluationDataset\Female'+str(candidate)+'\\Test1\\classe_%d.dat' % i, dtype=np.int16)
        data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
        dataset_example = format_data_to_train(data_read_from_file) 
        examples.append(dataset_example) 
        labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))

    examples, labels = shift_electrodes(examples, labels)
    list_dataset.append(examples)
    list_labels.append(labels)

print("Finished Reading Data from Test_1")
print("")
datasets = [list_dataset, list_labels]
np.save("NPY_files\loaded_Test_1.npy", datasets)

print("Done, Now run prepare_evaluation_dataset.py")
print("")

