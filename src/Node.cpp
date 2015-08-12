/* 
	Ma'ad Shipchandler
	Neural Net implementation file
	2-6-2015
*/

#include <iostream>
#include "Node.h"
//#include "functions.h"
#include "activation_functions.h"

// class Node functions ------------------------------------------------------------------------

Node::Node() : type(UNDEF), nodeID(UNDEF), nodeVal(UNDEF) { }

Node::Node(int _type, int _nodeID, double _nodeVal, bool _biasFlag, 
									iActivationFunction* _act_func)
{
	type = _type;
	nodeID = _nodeID;
	nodeVal = _nodeVal;
	biasFlag = _biasFlag;
	errorGradient = 0.0;
	act_func = _act_func;
}

// Should only be explicitly used for an Input Layer.
void Node::setNodeVal(double _nodeVal)
{
	nodeVal = _nodeVal; 
}

double Node::getNodeVal()
{
	return nodeVal;
}

// Set the weights of the connections.
void Node::setWeight(double _weight, Node* _destination)
{
	weights.push_back(NodeCxn(_weight, _destination));	
}

// Returns the weight of a particular connection (of the corresponding address)
double Node::getWeight(Node* address)
{
	for(int i = 0; i < (int)weights.size(); i++)
	{
		if(weights[i].destination == address)
			return weights[i].weight;
	}

	return ERROR; // Should never reach here if Net is set up properly.
}

void Node::setWeightPort(Node* address) 
{
	weight_port.push_back(address);
}

// Finds the value of the Node after calculating the weighted sum and passing it
// through the activation function. Used only for hidden and output layers.
void Node::calculateNodeVal()
{
	double weightedSum = 0.0;

	for(int i = 0; i < (int)weight_port.size(); i++)
	{
		weightedSum += weight_port[i]->getNodeVal() * weight_port[i]->getWeight(this);
	}

	//setNodeVal(fn::activationFunction_step(weightedSum));
	//setNodeVal(fn::activationFunction_sigmoid(weightedSum));
	//setNodeVal(fn::activationFunction_tanh(weightedSum));
	//setNodeVal(fn::activationFunction_linear(weightedSum));

	setNodeVal(act_func->activationFunction(weightedSum));
}

// Function to allow updating the weights during the learning. Updates the weights
// of a particular connection (of the corresponding address)
void Node::setWeight_forUpdate(double updatedWeight, Node* address)
{
	for(int i = 0; i < (int)weights.size(); i++)
	{
		if(weights[i].destination == address)
			weights[i].weight = updatedWeight;
	}
}

bool Node::isBias()
{
	return biasFlag;
}

// Function to calculate the errorGradient for the respective nodes.
void Node::calculateErrorGradients(double ideal_output)
{
	if(type == HIDDEN)
	{
		double weightedSum_errors = 0.0;
		for(int i = 0; i < (int)weights.size(); i++)
		{
			weightedSum_errors += weights[i].weight
										 * weights[i].destination->getErrorGradient();
		}

		// Gradient descent method
		//errorGradient = nodeVal * (1 - nodeVal) * (weightedSum_errors); // Derivative of sigmoid.
		//errorGradient = (1 - (tanh(nodeVal) * tanh(nodeVal))) * (weightedSum_errors); // Using the derivative of tanh.

		errorGradient = act_func->derivative(nodeVal) * (weightedSum_errors);
	}

	else if(type == OUTPUT)
	{
		// Gradient descent method
		//errorGradient = nodeVal * (1 - nodeVal) * (ideal_output - nodeVal); // Derivative of sigmoid.
		//errorGradient = (1 - (tanh(nodeVal) * tanh(nodeVal))) * (ideal_output - nodeVal); // Using the derivative of tanh.

		errorGradient = act_func->derivative(nodeVal) * (ideal_output - nodeVal);
	}

	else // if the type is INPUT or something other than HIDDEN AND OUTPUT.
	{
		std::cout << "SYSTEM MSG: Error: Cannot calculate errorGradient" 
					<< std::endl; // Should never reach here.
	}
}

double Node::getErrorGradient()
{
	return errorGradient;
}

// Functionality for updating weights. To be used after errorGradients have been calculated.
void Node::updateWeights()
{
	double deltaWeight = 0.0, updatedWeight = 0.0;
	for(int i = 0; i < (int)weight_port.size(); i++)
	{
		// If it is a bias node, node value doesn't come into play, becuase the node value is always 1, generally.
		if(weight_port[i]->isBias())
			deltaWeight = ALPHA * errorGradient;
		else
			deltaWeight = ALPHA * weight_port[i]->getNodeVal() * errorGradient;

		updatedWeight = weight_port[i]->getWeight(this) + deltaWeight;
		weight_port[i]->setWeight_forUpdate(updatedWeight, this);
	}
}

// Display node contents. Used for debugging.
void Node::display()
{
	std::cout << "----------------" << this << "----------------" << std::endl;
	std::cout << " type: " << type << std::endl;
	std::cout << " nodeID: " << nodeID << std::endl;
	std::cout << " nodeVal: " << nodeVal << std::endl;
	std::cout << " weight_port: " << std::endl;
		for (int i = 0; i < (int)weight_port.size(); ++i)
			std::cout << "  " << weight_port[i] << std::endl;
	std::cout << " weights: " << std::endl;
		for(int i = 0; i < (int)weights.size(); i++)
		{
			std::cout << "  " << weights[i].weight << 
						"-->" << weights[i].destination << std::endl;
		}
	std::cout << "-----------------------------------------" << std::endl;
}

// ---------------------------------------------------------------------------------------------

NodeCxn::NodeCxn(double _weight, Node* _destination) : weight(_weight), destination(_destination) { }