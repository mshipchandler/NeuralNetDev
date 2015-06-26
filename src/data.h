/* 
	Ma'ad Shipchandler
	Training / Testing data for the Neural Net
	3-6-2015
*/

/*
	Training data format:
		Inputs:
			Input A: 	 {Input X, Input X + 1, ... Input X + n}
			Input A + 1: {Input X, Input X + 1, ... Input X + n}
			...
			Input A + m: {Input X, Input X + 1, ... Input X + n}
		Outputs:
			Output A: 	  { Output X, Output X + 1, ... Output X + n}
			Output A + 1: { Output X, Output X + 1, ... Output X + n}
			...
			Output A + m: { Output X, Output X + 1, ... Output X + n}
*/

const std::vector<std::vector<double>> NAND_inputs = 
{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1}
};

const std::vector<std::vector<double>> NAND_outputs = 
{
	{1},
	{1},
	{1},
	{0}
};

const std::vector<std::vector<double>> NOR_inputs = 
{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1}
};

const std::vector<std::vector<double>> NOR_outputs = 
{
	{1},
	{0},
	{0},
	{0}
};

const std::vector<std::vector<double>> XOR_inputs = 
{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1}
};

const std::vector<std::vector<double>> XOR_outputs = 
{
	{0},
	{1},
	{1},
	{0}
};