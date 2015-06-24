/* 
	Ma'ad Shipchandler
	Neural Net activation function implementations
	11-6-2015
*/

#pragma once
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath> // exp(), tanh()

class iActivationFunction
{
	public:
		virtual double activationFunction(double) = 0;
		virtual double derivative(double) = 0;
		virtual ~iActivationFunction() = default;
};

/*
	Step activation functions are used in MCP node networks, that do not
	have the ability to learn. Weights have to be calculated by other means
	to use such a network. That is why this function may not be usefull for
	out MLP network. -MS
*/
class ActivationStep : public iActivationFunction
{
	public:
		// Outputs either 0 or 1.
		double activationFunction(double weightedSum)
		{
			if(weightedSum >= 0)
				return 1;
			else 
				return 0;
		}
		double derivative(double nodeVal)
		{
			// Pending. -MS
			return -1; // Error state.
		}
};

class ActivationSigmoid : public iActivationFunction
{
	public:
		// Outputs numbers in the range 0 to +1.
		double activationFunction(double weightedSum)
		{
			return (1 / (1 + exp(-1 * weightedSum))); 
		}
		double derivative(double nodeVal)
		{
			return nodeVal * (1 - nodeVal);
		}
};

class ActivationSigmoidBipolar : public iActivationFunction
{
	public:
		double activationFunction(double weightedSum)
		{
			return -1 + (2 / (1 + exp(-1 * weightedSum)));
		}
		double derivative(double nodeVal)
		{
			return 0.5 * (1 + nodeVal) * (1 - nodeVal);
		}
};

class ActivationTanh : public iActivationFunction
{
	public:
		// Outputs numbers in the range -1 to +1.
		double activationFunction(double weightedSum)
		{
			return tanh(weightedSum);
		}
		double derivative(double nodeVal)
		{
			return (1 - (tanh(nodeVal) * tanh(nodeVal)));
		}
};

class ActivationLinear : public iActivationFunction
{
	public:
		/*  Has no use theoretically and is essentially no activation
			at all, but can be used to output the entire range of numbers.
		*/
		double activationFunction(double weightedSum)
		{
			return weightedSum;
		}
		double derivative(double nodeVal)
		{
			std::cout << "SYSTEM MSG: Unusable derivative for linear activation function." << std::endl;
			return -1; // Error state.
		}
};

#endif