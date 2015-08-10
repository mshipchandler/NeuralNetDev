/* 
	Ma'ad Shipchandler
	Set up and Run the network implementation file (For unsupervised learning)
	6-7-2015
*/

#include <iostream>
#include <vector> // std::vector
#include <random>
#include <unistd.h> // usleep()
#include "Node.h"
#include "data.h"
#include "activation_functions.h"

#define INPUTNUM 2 // Not including the Bias for the hidden layer.
#define HIDDENNUM 3 // Not including the Bias for output the layer.
#define OUTPUTNUM 1

// Function will return a random weight between -1.0 and 1.0
double randomWeight()
{
	std::random_device rd;
	std::default_random_engine e1(rd());
	std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

	return uniform_dist(e1);
}

void compare(double real_output, double ideal_output)
{
	std::cout << "real_output: " << real_output << ", ideal_output: " 
								<< ideal_output << std::endl;

	//std::cin.ignore(); // For debugging
	//usleep(20000);
}

int main(int argc, char* argv[])
{
	std::vector<Node> inputLayer;
	std::vector<Node> hiddenLayer;
	std::vector<Node> outputLayer;
	int ID = 0;
	// Bias node activation is always 1, generally.
	double biasNodeForHidden_Val = 1.0, biasNodeForOutput_Val = 1.0;

	// Setup Layers -------------------------------------------------

	// DO NOT FORGET TO DEALLOCATE ----------------------------------
	//iActivationFunction* act_func = new ActivationStep();
	//iActivationFunction* act_func = new ActivationSigmoid();
	//iActivationFunction* act_func = new ActivationSigmoidBipolar();
	iActivationFunction* act_func = new ActivationTanh();
	//iActivationFunction* act_func = new ActivationLinear();
	// --------------------------------------------------------------

	for(int i = 0; i < INPUTNUM; i++)
		inputLayer.push_back(Node(INPUT, ID++, 0.0, false, act_func)); 
	inputLayer.push_back(Node(INPUT, ID++, biasNodeForHidden_Val, true, act_func)); // Bias Node

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer.push_back(Node(HIDDEN, ID++, 0.0, false, act_func));
	hiddenLayer.push_back(Node(INPUT, ID++, biasNodeForOutput_Val, true, act_func)); // Bias Node

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer.push_back(Node(OUTPUT, ID++, 0.0, false, act_func));

	// --------------------------------------------------------------

	// Setting up connections and randomizing weights ---------------

	for(int i = 0; i < HIDDENNUM; i++)
	{
		for(int j = 0; j < INPUTNUM + 1; j++) // + 1 for Bias Node
		{
			inputLayer[j].setWeight(randomWeight(), &hiddenLayer[i]);
			hiddenLayer[i].setWeightPort(&inputLayer[j]);
		}
	}

	for(int i = 0; i < OUTPUTNUM; i++)
	{
		for(int j = 0; j < HIDDENNUM + 1; j++) // + 1 for Bias Node
		{
			hiddenLayer[j].setWeight(randomWeight(), &outputLayer[i]);
			outputLayer[i].setWeightPort(&hiddenLayer[j]);
		}
	}

	// --------------------------------------------------------------

	int trainingCount = 0, training_set_num = 0, ideal_output;
	const std::vector<std::vector<double>> training_inputs = XOR_inputs, 
			training_outputs = XOR_outputs; 
	int training_inputs_size = training_inputs.size();
	// Running the neural net ---------------------------------------

	while(trainingCount < 5000)
	{
		for(int i = 0; i < INPUTNUM; i++) // Exclude the bias iteration because of the constant nodeVal.
			{
				inputLayer[i].setNodeVal(training_inputs[training_set_num][i]);
			}
			std::cout << "Inputs: A: " << training_inputs[training_set_num][0] << ", B: " 
										<< training_inputs[training_set_num][1] << std::endl;
			ideal_output = training_outputs[training_set_num][0];
			training_set_num++;

		if(training_set_num == training_inputs_size) { training_set_num = 0; }

		for(int i = 0; i < HIDDENNUM; i++) // Exclude the bias iteration because of the constant nodeVal.
		{
			hiddenLayer[i].calculateNodeVal();
			hiddenLayer[i].updateWeights_unsupervised();
		}

		for(int i = 0; i < OUTPUTNUM; i++)
		{
			outputLayer[i].calculateNodeVal();
			outputLayer[i].updateWeights_unsupervised();
		}

		compare(outputLayer.back().getNodeVal(), ideal_output);

		trainingCount++;
		std::cout << "Training Cycle: " << trainingCount << std::endl << std::endl;
	}

	// --------------------------------------------------------------



	return 0;
}