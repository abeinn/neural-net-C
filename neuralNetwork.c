#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

double init_weights() { return ((double) rand()) / ((double) RAND_MAX); }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double dSigmoid(double x) { return x * (1 - x); }

void shuffle(int *array, int n) {
    if (n > 1) {
        for (int i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(void) {

    srand(123);

    const double alpha = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];


    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                          {1.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 1.0f}}; 

    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}}; 

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++){
            hiddenWeights[i][j] = init_weights();
        }
    }
    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = init_weights();
        }
    }
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weights();
    }
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weights();
    }



    int numEpochs = 10000;

    int trainingSetOrder[] = {0, 1, 2, 3};

    for (int epoch = 0; epoch < numEpochs; epoch++) {

        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++) {
            
            int i = trainingSetOrder[x];

            // Hidden layer activation
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            
            // Output layer activation
            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g %g  Output: %g    Expected Output: %g \n",
                    training_inputs[i][0], training_inputs[i][1],
                    outputLayer[0], training_outputs[i][0]);

            // Backprop
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * alpha; 
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * alpha; 
                }
            }
            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * alpha; 
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * alpha; 
                }
            }
        }
    }
}