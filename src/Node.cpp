/* 
	Ma'ad Shipchandler
	Neural Net implementation file
	2-6-2015
*/

#include <iostream>
#include "Node.h"
#include "functions.h"

// class Node functions ------------------------------------------------------------------------

Node::Node() : type(UNDEF), nodeID(UNDEF), nodeVal(UNDEF) { }

Node::Node(int _type, int _nodeID, double _nodeVal)
{
	type = _type;
	nodeID = _nodeID;
	nodeVal = _nodeVal;
}

void Node::setWeight(double _weight, Node* _destination)
{
	weights.push_back(NodeCxn(_weight, _destination));	
}

double Node::getWeight(Node* address)
{
	for(int i = 0; i < weights.size(); i++)
	{
		if(weights[i].destination == address)
			return weights[i].weight;
	}

	return ERROR; // Should never reach here if Net is set up properly.
}

void Node::calculateNodeVal()
{
	double weightedSum = 0.0;

	for(int i = 0; i < weight_port.size(); i++)
	{
		weightedSum += weight_port[i]->getNodeVal() * weight_port[i]->getWeight(this);
	}

	setNodeVal(fn::activationFunction(weightedSum));
}

void Node::setWeight_forUpdate(double updatedWeight, Node* address)
{
	for(int i = 0; i < weights.size(); i++)
	{
		if(weights[i].destination == address)
			weights[i].weight = updatedWeight;
	}
}

void Node::calculateErrorGradients(double ideal_output)
{
	switch(type)
	{
		case INPUT:
			break;

		case HIDDEN:
		{
				double weightedSum_errors = 0.0;
				for(int i = 0; i < weights.size(); i++)
				{
					weightedSum_errors += weights[i].weight
										 * weights[i].destination->getErrorGradient();
				}

				errorGradient = nodeVal * (1 - nodeVal) * (weightedSum_errors);
			break;
		} // Used to restrict the scope of weightedSum_errors to case HIDDEN.

		case OUTPUT:
				errorGradient = nodeVal * (1 - nodeVal) * (ideal_output - nodeVal);
			break;

		default:
			break; // Should never reach here.
	}
}

void Node::updateWeights()
{
	double deltaWeight = 0.0, updatedWeight = 0.0;
	for(int i = 0; i < weight_port.size(); i++)
	{
		deltaWeight = ALPHA * weight_port[i]->getNodeVal() * errorGradient;
		updatedWeight = weight_port[i]->getWeight(this) + deltaWeight;
		weight_port[i]->setWeight_forUpdate(updatedWeight, this);
	}
}

void Node::display()
{
	std::cout << "----------------" << this << "----------------" << std::endl;
	std::cout << " type: " << type << std::endl;
	std::cout << " nodeID: " << nodeID << std::endl;
	std::cout << " nodeVal: " << nodeVal << std::endl;
	std::cout << " weight_port: " << std::endl;
		for (int i = 0; i < weight_port.size(); ++i)
			std::cout << "  " << weight_port[i] << std::endl;
	std::cout << " weights: " << std::endl;
		for(int i = 0; i < weights.size(); i++)
		{
			std::cout << "  " << weights[i].weight << 
						"-->" << weights[i].destination << std::endl;
		}
	std::cout << "-----------------------------------------" << std::endl;
}

// ---------------------------------------------------------------------------------------------