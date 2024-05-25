package main

import (
	"deep_learning/solver"
	"fmt"

	ngo "github.com/adynascimento/deep-learning/gonum"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/plot/plot"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// loading data
	time := ngo.Linspace(0.0, 39.99, 4000)
	data := solver.LoadFromFile("data.csv")
	derivativeData := solver.LoadFromFile("derivative.csv")

	// training dimension
	trainingDim := int(0.25 * float64(len(time)))

	//split data into training and testing dataset
	xTrain, xTest := solver.Split(data, 0.25)
	yTrain, yTest := solver.Split(derivativeData, 0.25)

	// input and output features
	inputDim := xTrain.RawMatrix().Rows
	outputDim := yTrain.RawMatrix().Rows

	// neural network model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{inputDim, 45, outputDim}, // neural network structure
		Activation:  network.TanhActivation,         // activation function
		Mode:        network.ModeRegression,         // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := neural.NewTrainer(network.TrainerConfig{
		Optimizer:        network.AdamOptimizer,
		LearningRate:     0.001,
		L2Regularization: 1.40e-06,
		NIterations:      20000,
	})
	model.Fit(xTrain, yTrain, true)
	fmt.Printf("training dataset error: %.6e\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing dataset error:  %.6e\n", model.Evaluate(xTest, yTest))

	// saves neural network model to file
	model.Save("neuralnetwork.json")

	// integrating the predictions
	x0 := mat.Col(nil, 0, xTrain)
	integratedPred := solver.SolveRK4(model, x0, len(time), time[1]-time[0])

	// mean squared error
	metric := ngo.Scale(1./float64(data.RawMatrix().Cols),
		ngo.Sum(ngo.Square(ngo.Sub(data, integratedPred)), ngo.OverColumns))
	fmt.Println("global integration error by feature:")
	fmt.Printf("%.6e\n", mat.Formatted(metric))

	// plotting
	plt := plot.NewPlot()
	plt.FigSize(12, 9)

	plt.Plot(time, mat.Row(nil, 0, data))
	plt.Plot(time, mat.Row(nil, 0, integratedPred))
	plt.Plot(ngo.Linspace(time[trainingDim], time[trainingDim], 10), ngo.Linspace(-2.0, 2.0, 10))
	plt.Title("neural network predictions")
	plt.XLabel("x values")
	plt.YLabel("y values")
	plt.Legend("analytical model", "model prediction", "end of training window")
	plt.XLim(0.0, 40.0)

	plt.Save("plot.png")
}
