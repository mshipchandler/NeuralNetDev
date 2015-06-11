/* 
	Ma'ad Shipchandler
	Neural Net misc. functions header file
	2-6-2015
*/

#pragma once
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath> // exp(), tanh()

namespace fn
{
	// Outputs either 0 or 1.
	double activationFunction_step(double weightedSum)
	{
		if(weightedSum >= 0)
			return 1;
		else
			return 0;
	}

	// Outputs numbers in the range 0 to +1.
	double activationFunction_sigmoid(double weightedSum)
	{
		return (1 / (1 + exp(-1 * weightedSum))); 
	}

	// Outputs numbers in the range -1 to +1.
	double activationFunction_tanh(double weightedSum)
	{
		return tanh(weightedSum);
	}

	/*  Has no use theoretically and is essentially no activation
		at all, but can be used to output the entire range of numbers.
	*/
	double activationFunction_linear(double weightedSum)
	{
		return weightedSum;
	}
}

#endif