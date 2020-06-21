# sEMG-classification
sEMG Hand Gestures Recognition Using Convolutional Neural Networks

# Required libraries: 

1.numpy
2.scipy
3.keras
4.json
5.os

# Structure

sMEG classification

	- Dataset
		- EvaluationDataset
		- TrainingDataset

	- Evaluation
		- NPY_files
		- evaluate_model.py
		- load_evaluation_dataset.py
		- prepare_Evaluation_Dataset.py

	- Training
		- 2_split_architecture
			-train_model.py
		- 4_split_architecture
		- NPY_files
		- load_training_dataset.py
		- prepare_training_dataset.py

	- README.txt


# How to launch ?

To Train Model

1.run Training/load_training_dataset.py
2.run Training/prepare_training_dataset.py
3.run Training/2_split_architecture/train_model.py


To evaluate_model

4.run Evaluation/load_evaluation_dataset.py
5.run Evaluation/prepare_evaluation_dataset.py

	#select a model you want to evaluate 

	a folder will be cretaed with the name of validation accuracy inside  
	"Training/2_split_architecture" everytime you train a mddel

	Eg: - Training
			- 2_split_architecture
				- 0.9425

6. assign the model folder name inside evaluate_model.py and run

	Eg: - val_acc=0.9425


7. You can find whole details about the model in data.txt inside model folder

	Eg: - Training
			- 2_split_architecture
				- 0.9425
					-data.txt



###### IMPORTANT #######

No need to load, prepare datasets every time, if you done it ones
you can direcly train your model and evaluate 

Enjoy!!!!
