<p align="left">Evaluating Classical Preprocessing vs. Quantum Circuit Complexity in Hybrid Models for Molecular Activity Prediction

Project structure

Data  
* raw
* processed

Models
* quantum_params.npy

Src  
* classical_model.py
* preprocess.py  
* quantum_data.py  
* quantum_data_reupload.py  
* quantum_model.py  
* quantum_model.reuploading  

First run preprocess.py
Then if you want to run the classical model, run that
If hybrid model, run quantum_model_reuploading.py, then run quantum_data_reupload.py


## preprocess.py
* Raw data is stored in data/raw
* Filter through the dataset and removes invalid molecules
* Convert to 2048 dimensional Circular Fingerprints
* Splits dataset into train, validation, and test splits using scaffold split
* Set the y labels to ints
* Select 50 negative and 50 positive cases from every split, those will become train, validation, and tets splits for hybrid model
* Scale data from I-pi, pi)
* Apply Principal Component Analysis to reduce dimensions to a specified amount compatible with hybrid model
* Edited data is stored in data/processed

## classical_model.py
XGBoost model
* Train the model
* Outputs Accuracy, AUC, and F1 scores

## quantum_model_reuploading.py
Because there aren't already made commands for traning  a hybrid model, we have to write the training loop ourselves
* Before training loop define Hamiltonian(for mean calculation), num_layers(number of layers), starting params(parameters), opt(optimizer), number of epochs
* In the traning loop
  * opt.step
    * Goes to run the loss function
      * To get the predictions, runs the variational_circuit
        * Split input into chunks equal to the number of qubits
        * Performs angle embedding every layer, every ansatz loop
        * Then calculates the mean of the expectation values of the qubits
      * Applies tanh to the predictions
      * Calculates the loss and returns the value
    * Updates parameters using Gradient Descent optimizer
  * calculates accuracy of train and validations and saves them
  * Saves parameters
* runs for 300 epochs
* At the end select parameters with highest validation accuracy
* Model parameters are saved in the Models folder

## quantum_data_reupload.py
* Run to get the accuracy, AUC and F1 scores of the hybrid model
