package main

import (
	"deep_learning/plot"
	"fmt"

	network "deep_learning/neuralNetwork"
	ngo "deep_learning/numeric"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {

	// loading data
	time, data := ngo.ReadData("data.txt")
	derivativeData := ngo.ReadDerivativeData("derivative.txt")

	// training dimension
	training_dim := int(0.25 * float64(len(time)))

	// training data
	inputData := mat.NewDense(len(data), training_dim, nil)
	outputData := mat.NewDense(len(derivativeData), training_dim, nil)
	for i := 0; i < inputData.RawMatrix().Rows; i++ {
		inputData.SetRow(i, data[i][:training_dim])
		outputData.SetRow(i, derivativeData[i][:training_dim])
	}

	// input and output features
	input_dim := inputData.RawMatrix().Rows
	output_dim := outputData.RawMatrix().Rows

	// hyperparameters
	nn_structure := []int{input_dim, 45, output_dim} // neural network structure
	activation_function := network.ActivationTanh    // activation function
	l2_regularization := 1.40e-06                    // regularization parameter
	num_iterations := 20000                          // number of iterations

	// neural network model
	model := network.NewNeuralNetwork(
		nn_structure,
		activation_function,
		l2_regularization,
		num_iterations,
	)

	// optimizer to train the model
	learning_rate := 0.001
	model.NewTrainer(network.AdamOptimizer, learning_rate)
	model.Fit(inputData, outputData, true)

	// saves neural network model to file
	model.Save("savedModel.json")

	// make predictions
	x0 := ngo.GetCol(inputData, 0)
	predictions := model.SolveRK4(x0, len(time), 0.01)

	// L2 error
	error := floats.Norm(ngo.SubSlices(data[0], predictions.RawRowView(0)), 2.0) / floats.Norm(data[0], 2.0)
	fmt.Printf("\nextrapolation error: %f %% \n", 100.0*error)

	// plotting
	plt := plot.NewPlot()
	plt.FigSize(12, 9)

	plt.Plot(time, data[0])
	plt.Plot(time, predictions.RawRowView(0))
	plt.Plot(ngo.Linspace(time[training_dim], time[training_dim], 10), ngo.Linspace(-2.0, 2.0, 10))
	plt.Title("neural network predictions")
	plt.XLabel("x values")
	plt.YLabel("y values")
	plt.Legend("true model", "prediction", "end of training window")
	plt.XLim(0.0, 40.0)

	plt.Save("plot.png")

}
