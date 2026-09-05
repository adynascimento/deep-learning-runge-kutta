package main

import (
	"fmt"
	"runge-kutta/solver"
	"strconv"
	"time"

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

	model := func(trialID int, params mlp.Params) float64 {
		nnStructure := []int{xTrain.RawMatrix().Cols}              // input dimension
		nnStructure = append(nnStructure, params.HiddenLayers...)  // hidden layers
		nnStructure = append(nnStructure, yTrain.RawMatrix().Cols) // output dimension

		// neural network model
		neural := mlp.NewNeuralNetwork(mlp.NeuralConfig{
			NNStructure: nnStructure,           // neural network structure
			Activation:  nncore.TanhActivation, // activation function
			Mode:        nncore.ModeRegression, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := neural.NewTrainer(mlp.TrainerConfig{
			Optimizer:    nncore.AdamOptimizer, // optimizer
			LearningRate: params.LearningRate,  // learning rate
			Epochs:       5000},                // number of iterations
			mlp.WithL2Regularization(params.L2Regularization),
			mlp.WithSeed(uint64(time.Now().UnixNano())),
		)
		model.Fit(xTrain, yTrain, mlp.WithVerbose(false))
		model.Save("./trials/model" + strconv.Itoa(trialID) + ".json")

		// make predictions and evaluate model
		return model.Evaluate(xTest, yTest)
	}

	study := mlp.NewHyperopt(mlp.SearchSpace{
		NHiddenLayersRange: mlp.IntRange{Min: 1, Max: 3},         // minimum and maximum number of layers
		NHiddenRange:       mlp.IntRange{Min: 30, Max: 80},       // minimum and maximum number of hidden units per layers
		LearningRateRange:  mlp.FloatRange{Min: 1e-4, Max: 1e-2}, // minimum and maximum of learning rate
		L2Range:            mlp.FloatRange{Min: 1e-6, Max: 1e-2}, // minimum and maximum of regularization parameter
		NTrials:            3,                                    // number of models
	})

	study.Optimize(hyperopt.Bayesian, hyperopt.Minimize, model)
	fmt.Println("best params:", study.GetBestParams())
}
