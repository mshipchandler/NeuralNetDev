/* 
	Ma'ad Shipchandler
	Neural Net misc. functions header file
	2-6-2015
*/

#pragma once
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath> // exp()

namespace fn
{
	double activationFunction(double weightedSum)
	{
		return (1 / (1 + exp(-1 * weightedSum))); 
	}
}

#endif