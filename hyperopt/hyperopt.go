package main

import (
	"fmt"
	"runge-kutta/solver"
	"strconv"

	"github.com/adynascimento/deep-learning/hyperopt"
	"github.com/adynascimento/deep-learning/mlp"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
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
		neural := mlp.NewNeuralNetwork(mlp.NeuralConfig{
			NNStructure: params.NNStructure,    // neural network structure
			Activation:  nncore.TanhActivation, // activation function
			Mode:        nncore.ModeRegression, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := neural.NewTrainer(mlp.TrainerConfig{
			Optimizer:    nncore.AdamOptimizer, // optimizer
			LearningRate: params.LearningRate,  // learning rate
			Epochs:       5000},                // number of iterations
			mlp.WithL2Regularization(params.L2Regularization),
			mlp.WithSeed(42),
		)
		model.Fit(xTrain, yTrain, mlp.WithVerbose(false))
		model.Save("./trials/model" + strconv.Itoa(trialID) + ".json")

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

	study.BayesianOptimization(hyperopt.Minimize, neuralNetworkModel)
	fmt.Println("best params:", study.GetBestParams())
}
