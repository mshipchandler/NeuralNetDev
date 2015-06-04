/*
	Ma'ad Shipchandler
	Activation Function tests
	4-6-2015
*/

#include <iostream>
#include <cmath>

double activationFunction_sigmoid(double x)
{
	return (1 / (1 + exp(-1 * x)));
}

double activationFunction_step(double x)
{
	if(x >= 0)
		return 1;
	else
		return 0;
}

int main(int argc, char* argv[])
{
	double x;
	std::cout << "Enter a value to run through an activation function: ";
	std::cin >> x;

	std::cout << "Running through sigmoid function: " 
			<< activationFunction_sigmoid(x) << std::endl;
	std::cout << "Running through step function: " 
			<< activationFunction_step(x) << std::endl;

	return 0;
}