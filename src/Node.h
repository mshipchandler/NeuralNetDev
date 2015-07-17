/* 
	Ma'ad Shipchandler
	Neural Net header file
	2-6-2015
*/

#pragma once
#ifndef NODE_H
#define NODE_H

#include <vector> // std::vector

#define INPUT 0
#define HIDDEN 1
#define OUTPUT 2
#define UNDEF -1
#define ERROR -1
#define ALPHA 0.1 // Learning rate
/*
	In practice, the learning rate is typically given a value of 0.1 or less.
	Higher values may provide faster convergence on a solution, but may also 
	increase instability and may lead to a failure to converge.
*/

class iActivationFunction; // Forward declaration for iActivationFunction* act_func
struct NodeCxn; // Forward declaration for std::vector<NodeCxn> weights
class Node
{
	int type;
	int nodeID;
	double nodeVal;
	bool biasFlag; // 'true' if bias node, 'false' otherwise.
	double errorGradient;
	iActivationFunction* act_func;
	std::vector<NodeCxn> weights;
	std::vector<Node*> weight_port; // Used only for HIDDEN and OUTPUT layers.

	public:
		Node();
		Node(int, int, double, bool, iActivationFunction*);

		// Setup and misc. functions
		void setNodeVal(double _nodeVal) { nodeVal = _nodeVal; } // Should only be explicitly used for an Input Layer.
		double getNodeVal() { return nodeVal; }
		double getNodeID() { return nodeID; }
		void setWeight(double, Node*);
		double getWeight(Node*);
		void setWeightPort(Node* address) { weight_port.push_back(address); }
		void calculateNodeVal();
		void setWeight_forUpdate(double, Node*);
		bool isBias() { return biasFlag; }

		// Function for the training cycle (SUPERVISED)
		void calculateErrorGradients(double);
		double getErrorGradient() { return errorGradient; }
		void updateWeights();

		// Functions for the training cycle (UNSUPERVISED)
		void updateWeights_unsupervised();

		// Debug Functions
		void display();

};

struct NodeCxn
{
	double weight;
	Node* destination;

	NodeCxn(double _weight, Node* _destination) : weight(_weight), destination(_destination) { }
};

#endif