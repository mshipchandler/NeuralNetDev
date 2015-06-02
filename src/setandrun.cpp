/* 
	Ma'ad Shipchandler
	Set up and Run the network implementation file
	2-6-2015
*/

#include <iostream>
#include <vector> // std::vector
#include "Node.h"

#define INPUTNUM 2
#define HIDDENNUM 3
#define OUTPUTNUM 1

double randomWeight()
{
	return 9.9;
}

int main(int argc, char* argv[])
{
	std::vector<Node> inputLayer;
	std::vector<Node> hiddenLayer;
	std::vector<Node> outputLayer;
	int ID = 0;
	double data[10][2];

	// Setup Layers -------------------------------------------------

	/*  Taking some values as initial nodeVal for all layers.
		inputLayer nodeVal will be replaced by training/testing data.
		hiddenLayer and outputLayer nodeVal will be calculated using
		weighted sums as per neural net heuristic. */

	for(int i = 0; i < INPUTNUM; i++)
		inputLayer.push_back(Node(INPUT, ID++, 4.2)); 

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer.push_back(Node(HIDDEN, ID++, 0.0));

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer.push_back(Node(OUTPUT, ID++, 0.0));
	
	// --------------------------------------------------------------

	// Setting up connections and randomizing weights ---------------

	for(int i = 0; i < HIDDENNUM; i++)
	{
		for(int j = 0; j < INPUTNUM; j++)
		{
			inputLayer[j].setWeight(randomWeight(), &hiddenLayer[i]);
			hiddenLayer[i].setWeightPort(&inputLayer[j]);
		}
	}

	for(int i = 0; i < OUTPUTNUM; i++)
	{
		for(int j = 0; j < HIDDENNUM; j++)
		{
			hiddenLayer[j].setWeight(randomWeight(), &outputLayer[i]);
			outputLayer[i].setWeightPort(&hiddenLayer[j]);
		}
	}

	// --------------------------------------------------------------

	// Assigning / calculating nodeVal for each node ----------------

	/*  Here, for the inputLayer, nodeVal is taken from the data set
		and for the hiddenLater and the outputLayer, nodeVal is 
		calculated usingweighted sums as per neural net heuristic. */

	for(int i = 0; i < INPUTNUM; i++)
	{

	}

	for(int i = 0; i < HIDDENNUM; i++)
	{
		hiddenLayer[i].calculateNodeVal();
	}

	for(int i = 0; i < OUTPUTNUM; i++)
	{
		outputLayer[i].calculateNodeVal();	
	}

	// --------------------------------------------------------------

	// Display Node information -------------------------------------

	for(int i = 0; i < INPUTNUM; i++)
		inputLayer[i].display();

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer[i].display();

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer[i].display();
	
	// --------------------------------------------------------------

	return 0;
}