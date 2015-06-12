/* 
	Ma'ad Shipchandler
	Neural Net activation function implementations
	11-6-2015
*/

#pragma once
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath> // exp(), tanh()

class ActivationFunction
{
	public:
		virtual double activationFunction(double) = 0;
		virtual double derivative(double) = 0;
};

class ActivationStep : public ActivationFunction
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
		}
};

class ActivationSigmoid : public ActivationFunction
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

class ActivationTanh : public ActivationFunction
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

class ActivationLinear : public ActivationFunction
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
			std::cout << "SYSTEM MSG: No derivative for linear activation function." << std::endl;
		}
};

#endif