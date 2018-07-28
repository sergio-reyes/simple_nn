#define INPUTNUM 16
#define HIDDENNUM 128
#define OUTPUTNUM 3
#define LEARNINGRATE 0.90

class NeuralNet
{
	public:
		//constructor
		NeuralNet();
	
		//sets weights using Nguyen-Widrow Initialization
		void SetWeights();

		//sets weights using a text file
		void SetWeights(char []);
	
		//go forward through the Neural Net
		void FeedForward(int []);

		//returns the results of the Net
		void GetResults(double []);

		//does the gradient descent correction to the net
		void Backpropagation(int []);

		//saves the weights to a text file	
		void SaveWeights(char []);

		//determines which output node is on
		int AnswerNode();
	
		//determines if a network has converged given an epsilon
		int Converged(double);


	private:
		
		//initializes weights w/ a value between -0.5 and 0.5	
		void InitializeWeights();
		
		//random number generator
		double Random(double,double);

		//get the norm of a given matrix
		double Norm(int);

		//Acitvation Function (--Binary Sigmoid Function--)	
		double f(double);

		//Derivative of Activation Function
		double fPrime(double);

		//broadcasts values to the hidden layer
		void BroadcastHidden(int);

		//broadcasts values to the output layers
		void BroadcastOutput(int);

		//descends to the hidden layer to figure it's adjustments
		void HiddenDescent(int []);

		//determines gradient between hidden and output links
		void SetHiddenGradient(int,int);

		//finds the delta value for the hidden nodes' weights
		void SetHiddenWeightAdjustment(int);
	
		//descends to the input layer to figure it's adjustments
		void InputDescent();

		//determines the gradient between the input and hidden links
		void SetInputGradient(int);

		//finds the delta value for the input nodes' weights
		void SetInputWeightAdjustment(int);

		//updates weights of links between layers
		void Update();

		//sums all of the weights in the network
		void NetWeightSummation();

		//structure for nodes in input layer
		struct OutputNode
		{
			double value;
		};
			
		//structure for nodes in input layer	
		struct InputNode
		{
			double value;
			//weight from input to hidden node
			double weight[HIDDENNUM];
			double oldWeight[HIDDENNUM];
		};

		//structure for nodes in hidden layer
		struct HiddenNode
		{
			double value;
			//weight from hidden node to output node
			double weight[OUTPUTNUM];
			double oldWeight[OUTPUTNUM];
		};

		//actual layers
		InputNode inputLayer[INPUTNUM + 1];
		HiddenNode hiddenLayer[HIDDENNUM + 1];
		OutputNode outputLayer[OUTPUTNUM];

		//correction gradients
		double hiddenGradient[OUTPUTNUM];
		double inputGradient[HIDDENNUM + 1];

		//sum of all of the weight in the network
		double oldNetWeight;

};  //end of class NeuralNet
