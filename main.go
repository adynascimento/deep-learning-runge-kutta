package main

import (
	"fmt"
	"runge-kutta/solver"

	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/plot/plotter"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// loading data
	time := ngo.Linspace(0.0, 39.99, 4000)
	data := solver.LoadFromFile("solver/dataset/data.csv")
	derivativeData := solver.LoadFromFile("solver/dataset/derivative.csv")

	// training dimension
	trainingDim := int(0.25 * float64(len(time)))

	//split data into training and testing dataset
	xTrain, xTest := ngo.Split(data, 0.25)
	yTrain, yTest := ngo.Split(derivativeData, 0.25)

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
		Optimizer:    network.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       20000},
		network.WithL2Regularization(1.40e-06))
	model.Fit(xTrain, yTrain, true)
	fmt.Printf("training dataset error: %.6e\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing dataset error:  %.6e\n", model.Evaluate(xTest, yTest))

	// temporal integration for predictions
	integratedPred := solver.SolveRK4(solver.Parameters{
		Func: model.Predict,
		X0:   mat.Col(nil, 0, xTrain),
		Tmax: time[len(time)-1],
		Step: time[1] - time[0],
	})

	// mean squared error
	metric := ngo.Scale(1./float64(data.RawMatrix().Cols),
		ngo.Sum(ngo.Square(ngo.Sub(data, integratedPred)), ngo.OverColumns))
	fmt.Println("global integration error by feature:")
	fmt.Printf("%.6e\n", mat.Formatted(metric))

	// plotting
	plt := plotter.NewSubplot(1, 2)
	plt.FigSize(23, 10)

	subplt := plt.Subplot(0, 0)
	subplt.Plot(time, mat.Row(nil, 0, data))
	subplt.Plot(time, mat.Row(nil, 0, integratedPred))
	subplt.Plot(ngo.Linspace(time[trainingDim], time[trainingDim], 10), ngo.Linspace(-2.0, 2.0, 10))
	subplt.Title("neural network predictions")
	subplt.XLabel("t values")
	subplt.YLabel("x1")
	subplt.Legend("analytical model", "model prediction", "end of training window")
	subplt.XLim(0.0, 40.0)
	subplt.Grid()

	subplt = plt.Subplot(0, 1)
	subplt.Grid()
	subplt.Plot(time, mat.Row(nil, 1, data))
	subplt.Plot(time, mat.Row(nil, 1, integratedPred))
	subplt.Plot(ngo.Linspace(time[trainingDim], time[trainingDim], 10), ngo.Linspace(-2.0, 2.0, 10))
	subplt.Title("neural network predictions")
	subplt.XLabel("t values")
	subplt.YLabel("x2")
	subplt.Legend("analytical model", "model prediction", "end of training window")
	subplt.XLim(0.0, 40.0)

	plt.Show()
	plt.Save("plot.png")
}
