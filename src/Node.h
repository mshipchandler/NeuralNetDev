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
#define ALPHA 0.45 // Learning rate

struct NodeCxn; // Forward declaration for std::vector<NodeCxn> weights
class Node
{
	int type;
	bool biasFlag;
	int nodeID;
	double nodeVal;
	double errorGradient;
	std::vector<NodeCxn> weights;
	std::vector<Node*> weight_port; // Used only for HIDDEN and OUTPUT layers.

	public:
		Node();
		Node(int, int, double, bool);

		// Setup and misc. functions
		void setNodeVal(double _nodeVal) { nodeVal = _nodeVal; } // Should only be explicitly used for an Input Layer.
		double getNodeVal() { return nodeVal; }
		void setWeight(double, Node*);
		double getWeight(Node*);
		void setWeightPort(Node* address) { weight_port.push_back(address); }
		void calculateNodeVal();
		void setWeight_forUpdate(double, Node*);
		bool isBias() { return biasFlag; }

		// Function for learning
		void calculateErrorGradients(double);
		double getErrorGradient() { return errorGradient; }
		void updateWeights();

		// Debug Functions
		void display();

};

struct NodeCxn
{
	double weight;
	double deltaWeight;
	Node* destination;

	NodeCxn(double _weight, Node* _destination) : weight(_weight), destination(_destination) { }
};

#endif