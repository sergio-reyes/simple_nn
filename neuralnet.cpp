#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "neuralnet.h"

using std::ofstream;
using std::ifstream;
using std::endl;

//constructor:
NeuralNet::NeuralNet()
{
	//variable declaration
	int index;

	//initialize the input layer
	inputLayer[0].value = 1;
	for (index = 1; index < INPUTNUM + 1; index++)
		inputLayer[index].value=0;

	//initialize the hidden layer
	hiddenLayer[0].value = 1;
	for (index = 1; index < HIDDENNUM + 1; index ++)
		hiddenLayer[index].value=0;

	//initialize the output layer
	for (index = 0; index < OUTPUTNUM; index++)
		outputLayer[index].value = 0;

	//initialize the weigts
	InitializeWeights();

}//end of constructor

//************************************************************************************************
void NeuralNet::InitializeWeights()
{
	//variable declaration
	int index;
	int inner;

	//set the weights from (InputLayer-->HiddenLayer)
	for (index = 0; index < INPUTNUM + 1; index ++) 
		for (inner = 0; inner < HIDDENNUM; inner ++)
		{
			inputLayer[index].weight[inner] = Random(-0.5,0.5);
			inputLayer[index].oldWeight[inner]=inputLayer[index].weight[inner];
		}

	//set the weights from (HiddenLayer-->OutputLayer)
	for (index = 0; index < HIDDENNUM + 1; index ++)
		for (inner = 0; inner < OUTPUTNUM; inner ++) 
			hiddenLayer[index].weight[inner] = Random(-0.5,0.5);

}//end of InitializeWeights

//************************************************************************************************

double NeuralNet::Random(double smallestNum, double largestNum)
{
	//variable declaration
	double randomNumber;

	//generate a random divisor
	randomNumber = double(random());

	//randomly select if the divisor is odd or even
	if (int(randomNumber) % 2 == 0 )
		randomNumber = randomNumber * -1;

	//get a value in between the parameters passed
	do
	{
		//calculate a random decimal number
		randomNumber = randomNumber / 10;

	} while ((randomNumber > largestNum) || (randomNumber < smallestNum));

	return(randomNumber);

}
//************************************************************************************************

void NeuralNet::SetWeights()
{
	//variable declaration
	double scaleFactor;
	int index;
	int inner;

	//calculate the scale factor
	scaleFactor = pow(HIDDENNUM,(1/INPUTNUM));
	scaleFactor = scaleFactor * (0.7);

	//set weights between (InputLayer-->HiddenLayer)
	for (index = 0; index < INPUTNUM+1; index ++)
		for (inner = 0; inner < HIDDENNUM; inner ++)
			inputLayer[index].weight[inner] = (scaleFactor * inputLayer[index].oldWeight[inner]) / (Norm(inner));	

	//set the weights for the first bias node in InputLayer
	for (index = 0; index < HIDDENNUM; index ++)
		inputLayer[0].weight[index] = Random((scaleFactor -1), scaleFactor);
	
}//end of SetWeights

//************************************************************************************************

double NeuralNet::Norm(int hiddenNodeIndex)
{
	//variable declaration
	double sum = 0;
	int index;
	
	//add the square of each vector element
	for (index = 0; index < INPUTNUM; index ++)
		sum = sum + (pow(inputLayer[index].weight[hiddenNodeIndex],2));

	//get the square root of the sum
	return(sqrt(sum));

}//end of Norm
//************************************************************************************************

void NeuralNet::FeedForward(int inputData[])
{
	//variable declaration
	int index;

	//set the bias in the inputLayer
	inputLayer[0].value = 1;

	//feed the values to the input layer
	for (index = 1; index < INPUTNUM; index ++)
		inputLayer[index].value = inputData[index-1];

	//calculate the values for each hidden node:
	hiddenLayer[0].value = 1;
	for (index = 1; index < HIDDENNUM; index ++)
		//broadcast to the hidden layer
		BroadcastHidden(index -1 );

	//calculate the values for the output nodes
	for (index = 0; index < OUTPUTNUM; index ++)
		//broadcast to the output layer
		BroadcastOutput(index);

}//end of FeedForward
//************************************************************************************************

double NeuralNet::f(double x)
{
	//BINARY SIGMOID FUNCTION

	//variable declaration
	double denominator;

	//calculate the function:	
	//f(x) = 1/[1+e^(-x)]

	denominator = exp(x * -1);
	denominator ++;

	//return th evalue
	return(1/denominator);

}//end of f(x)
//************************************************************************************************

double NeuralNet::fPrime(double x)
{
	//variable declaration
	double result;

	//calculate the function:
	//f'(x) = f(x)[1-f(x)]

	result = f(x);
	result = result * (1 - result);

	//return the answer
	return(result);

}//end of fPrime
//************************************************************************************************

void NeuralNet::BroadcastHidden(int hiddenNode)
{
	//variable declaration
	int index;
	double sum = 0;

	//sumation of all the input node values * lin to hidden node
	for (index = 0; index < INPUTNUM + 1; index ++ )
		sum = sum + (inputLayer[index].value * inputLayer[index].weight[hiddenNode]);

	//find the value for each node in the hidden layer
	hiddenLayer[hiddenNode].value = f(sum);

}//end of BroadcastHidden
//************************************************************************************************

void NeuralNet::BroadcastOutput(int outputNode)
{
	//variable declaration
	int index;
	double sum = 0;

	//summation of all the hidden node values * weight to output node
	for (index = 0; index < HIDDENNUM + 1; index ++)
		sum = sum + (hiddenLayer[index].value * hiddenLayer[index].weight[outputNode]);

	//set the value of the output node
	outputLayer[outputNode].value = f(sum);

}//end of BroadcastOutput	
//************************************************************************************************

void NeuralNet::GetResults(double results[])
{
	//variable declaratoin
	int index;

	for ( index = 0; index < OUTPUTNUM; index ++)
		results[index] = outputLayer[index].value;

}//end of GetResults
//************************************************************************************************

void NeuralNet::Backpropagation(int target[])
{
	//calculate the sum of the old weights
	NetWeightSummation();

	//descend to the hidden layer
	HiddenDescent(target);

	//descend to th einput layer
	InputDescent();

	//update Weights and biases	
	Update();

}//end of Backpropagation
//************************************************************************************************

void NeuralNet::HiddenDescent(int target[])
{
	//variable declaration
	int index;

	//determine the gradient for the weights between hidden & output
	for (index = 0; index < OUTPUTNUM; index ++)
		SetHiddenGradient(index,target[index]);

	//calculate weight correction for each hidden node
	for(index = 0; index < HIDDENNUM + 1; index ++)
		SetHiddenWeightAdjustment(index);

}//end of HiddenDescent	

//************************************************************************************************

void NeuralNet::SetHiddenGradient(int outputNode, int targetValue)
{

	//variable declaration
	int index;
	double sum = 0;

	//summation of all hte hidden node values * weight of links
	for (index = 0; index < HIDDENNUM + 1; index ++)
		sum = sum + (hiddenLayer[index].value * hiddenLayer[index].weight[outputNode]);

	//calculate the gradient
	hiddenGradient[outputNode] = (targetValue - outputLayer[outputNode].value) * fPrime(sum);

}//SetHiddenGradient
//************************************************************************************************

void NeuralNet::SetHiddenWeightAdjustment(int hiddenNode)
{

	//variable declaration
	int index;

	//find the delta correction ofr each weight
	for (index = 0; index < OUTPUTNUM; index ++)
		hiddenLayer[hiddenNode].oldWeight[index] = LEARNINGRATE * hiddenGradient[index] * hiddenLayer[hiddenNode].value;

}// end of SetHiddenWeightAdjustment
//************************************************************************************************

void NeuralNet::InputDescent()
{
	//variable declaration
	int index;

	//determine the gradient for the weights between Input and Hidden
	for (index = 0; index < HIDDENNUM + 1; index ++)
		SetInputGradient(index);

	//clculae weight correction for each input node
	for (index = 0; index < INPUTNUM + 1; index ++)
		SetInputWeightAdjustment(index);

}//end of InputDescent
//************************************************************************************************

void NeuralNet::SetInputGradient(int hiddenNode)
{
	//variable declaration
	int index;
	double sum;

	//initialization
	inputGradient[hiddenNode] = 0;
	sum = 0;

	//calculate the gradient for each hidden node
	for (index = 0; index < OUTPUTNUM; index ++)
		inputGradient[hiddenNode] = inputGradient[hiddenNode] \
	 	+ (hiddenGradient[index] * hiddenLayer[hiddenNode].weight[index]);

	//calcuate: S(weight * limk) from input to hidden
	for (index = 0; index < INPUTNUM + 1; index ++)
		sum = sum + (inputLayer[index].value * inputLayer[index].weight[hiddenNode]);

	//put the gradient through f'(x)
	inputGradient[hiddenNode] = inputGradient[hiddenNode] * fPrime(sum);

}//end of SetInputGradient
//************************************************************************************************

void NeuralNet::SetInputWeightAdjustment(int inputNode)
{

	//variable declaration
	int index;

	//find the delta correction for each weight
	for (index = 0; index < HIDDENNUM; index ++)
		inputLayer[inputNode].oldWeight[index] = LEARNINGRATE * inputGradient[index] * inputLayer[inputNode].value;
	
}// end of SetInputWeightAdjustment
//************************************************************************************************

void NeuralNet::Update()
{

	//variable declaration
	int node;
	int link;

	//update the weights between the hidden and output layers
	for (node = 0; node < HIDDENNUM + 1; node ++)
		for (link = 0; link < OUTPUTNUM; link ++)
			hiddenLayer[node].weight[link] = hiddenLayer[node].weight[link] + hiddenLayer[node].oldWeight[link];
	
	//update the weights between the input and hidden layer
	for (node = 0; node < INPUTNUM + 1; node ++)
		for (link = 0; link < HIDDENNUM; link ++)
			inputLayer[node].weight[link] = inputLayer[node].weight[link] + inputLayer[node].oldWeight[link];

}//end of Update
//************************************************************************************************

void NeuralNet::SaveWeights(char fileName[])
{
	//variable declaration
	ofstream file(fileName);
	int node;
	int link;

	//save the weights of the input layer
	for (node = 0; node < INPUTNUM + 1; node ++)
		for (link = 0; link < HIDDENNUM; link ++)
		file << inputLayer[node].weight[link] << endl;

	//save the weights of the hidden layer
	for (node = 0; node < HIDDENNUM + 1; node ++)
		for (link= 0; link < OUTPUTNUM; link ++)
			file << hiddenLayer[node].weight[link] << endl;	

	//close the text file
	file.close();

}//end of SaveWeights
//************************************************************************************************

void NeuralNet::SetWeights(char fileName[])
{
	//variable declaration
	int node;
	int link;

	//open up the text file
	ifstream file(fileName);

	//get the weights for the input layer
	for (node = 0; node < INPUTNUM + 1; node++)
		for (link = 0; link < HIDDENNUM; link ++)
			file >> inputLayer[node].weight[link];

	//get the weights for the hidden layer
	for (node =0; node < HIDDENNUM + 1; node ++)
		for (link =0; link < OUTPUTNUM; link ++)
			file >> hiddenLayer[node].weight[link];

	//close the text file
	file.close();

}//end of SetWeights
//************************************************************************************************

int NeuralNet:: AnswerNode()
{

	//variable declaration
	int key;
	int index;

	//initialize 
	key=0;

	//go through the output nodes
	for (index = 1; index < OUTPUTNUM; index ++)
		if (outputLayer[index].value > outputLayer[key].value)
			key=index;

	//return the value of the node w/ highest value
	return(key);

}//end of AnswerNode
//************************************************************************************************

int NeuralNet::Converged(double epsilon)
{
	//variable declaration
	int node;
	int link;
	double sum;
	double delta;

	//initialize
	sum=0;

	//sum all of the weights between hidden and output layers
	for (node = 0; node < HIDDENNUM + 1; node ++)
		for (link = 0; link < OUTPUTNUM; link ++)
			sum += hiddenLayer[node].weight[link];

	//sum all of the weights between the input and hidden layers 
	for (node = 0; node < INPUTNUM + 1; node ++)
		for (link = 0; link < HIDDENNUM; link ++)
			sum += inputLayer[node].weight[link];

	//find delta of weights between old and new network weights
	delta = oldNetWeight - sum;

	//get the absolute value of delta
	if (delta < 0)
		delta *= -1;

	//determine whether or not to converge
	if (delta < epsilon)
		return(1);
	else
		return(0);

}//end of Converged
//************************************************************************************************

void NeuralNet::NetWeightSummation()
{
	//variable declaration
	int node;
	int link;

	//initialize	
	oldNetWeight = 0;

	//add the weights between hidden and output layer
	for (node = 0; node < HIDDENNUM + 1; node ++)
		for (link = 0; link < OUTPUTNUM; link ++)
			oldNetWeight += hiddenLayer[node].weight[link];

	//add the weights between input and hidden layer
	for (node = 0; node < INPUTNUM + 1; node ++)
		for (link = 0; link < HIDDENNUM; link ++)
			oldNetWeight += inputLayer[node].weight[link];

}//end of NetWeightSummation
//************************************************************************************************

