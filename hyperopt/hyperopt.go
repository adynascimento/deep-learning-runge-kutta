package main

import (
	"fmt"
	"runge-kutta/solver"
	"strconv"

	"github.com/adynascimento/deep-learning/hyperopt"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
)

func main() {
	// loading data
	data := solver.LoadFromFile("../solver/dataset/data.csv")
	derivativeData := solver.LoadFromFile("../solver/dataset/derivative.csv")

	//split data into training and testing dataset
	xTrain, xTest := ngo.Split(data, 0.25)
	yTrain, yTest := ngo.Split(derivativeData, 0.25)

	neuralNetworkModel := func(trialID int, params hyperopt.Params) float64 {
		// neural network model
		neural := network.NewNeuralNetwork(network.NeuralConfig{
			NNStructure: params.NNStructure,     // neural network structure
			Activation:  network.TanhActivation, // activation function
			Mode:        network.ModeRegression, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := neural.NewTrainer(network.TrainerConfig{
			Optimizer:        network.AdamOptimizer,   // optimizer
			LearningRate:     params.LearningRate,     // learning rate
			L2Regularization: params.L2Regularization, // l2 regularization
			NIterations:      20000,                   // number of iterations
		})
		model.Fit(xTrain, yTrain, true)
		model.Save("./trials/networkmodel" + strconv.Itoa(trialID) + ".json")

		// make predictions and evaluate model
		return model.Evaluate(xTest, yTest)
	}

	study := hyperopt.NewHyperparameterOptimization(
		hyperopt.SearchSpace{
			InputDim:          xTrain.RawMatrix().Rows,
			OutputDim:         yTrain.RawMatrix().Rows,
			NLayersRange:      []int{3, 5},           // minimum and maximum number of layers
			NHiddenRange:      []int{30, 80},         // minimum and maximum number of hidden units per layers
			LearningRateRange: []float64{1e-4, 1e-2}, // minimum and maximum of learning rate
			LambdRange:        []float64{1e-6, 1e-2}, // minimum and maximum of regularization parameter
			NModels:           3,                     // number of models
		})

	study.RandomSearchOptimization(hyperopt.Minimize, neuralNetworkModel)
	fmt.Println("best params:", study.GetBestParams())
}
