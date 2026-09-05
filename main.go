package main

import (
	"fmt"
	"runge-kutta/solver"
	"time"

	"github.com/adynascimento/deep-learning/mlp"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
	"github.com/adynascimento/plot/plotter"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// loading data
	times := ngo.Linspace(0.0, 39.99, 4000)
	data := solver.LoadFromFile("solver/dataset/data.csv")
	derivativeData := solver.LoadFromFile("solver/dataset/derivative.csv")

	// training dimension
	trainingDim := int(0.25 * float64(len(times)))

	//split data into training and testing dataset
	xTrain, xTest := ngo.Split(data, 0.25)
	yTrain, yTest := ngo.Split(derivativeData, 0.25)

	// input and output features
	inputDim := xTrain.RawMatrix().Cols
	outputDim := yTrain.RawMatrix().Cols

	// neural network model
	neural := mlp.NewNeuralNetwork(mlp.NeuralConfig{
		NNStructure: []int{inputDim, 45, outputDim}, // neural network structure
		Activation:  nncore.TanhActivation,          // activation function
		Mode:        nncore.ModeRegression,          // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := neural.NewTrainer(mlp.TrainerConfig{
		Optimizer:    nncore.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       10000},
		mlp.WithBatchSize(xTrain.RawMatrix().Rows),
		mlp.WithL2Regularization(1.40e-05),
		mlp.WithSeed(uint64(time.Now().UnixNano())),
	)
	model.Fit(xTrain, yTrain, mlp.WithLogInterval(1000))
	model.Save("model.json")

	fmt.Printf("training dataset error: %.6e\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing dataset error:  %.6e\n", model.Evaluate(xTest, yTest))

	// temporal integration for predictions
	integratedPred := solver.SolveRK4(solver.Parameters{
		Func: model.Predict,
		X0:   xTrain.RawRowView(0),
		Tmax: times[len(times)-1],
		Step: times[1] - times[0],
	})

	// mean squared error
	metric := ngo.Scale(1./float64(data.RawMatrix().Rows), ngo.Sum(ngo.Square(ngo.Sub(data, integratedPred)), ngo.OverRows))
	fmt.Println("global integration error by feature:")
	fmt.Printf("%.6e\n", mat.Formatted(metric))

	// plotting
	plt := plotter.NewSubplot(1, 2)
	plt.FigSize(23, 10)

	subplt := plt.Subplot(1, 1)
	subplt.Plot(times, mat.Col(nil, 0, data))
	subplt.Plot(times, mat.Col(nil, 0, integratedPred))
	subplt.Plot(ngo.Linspace(times[trainingDim], times[trainingDim], 10), ngo.Linspace(-2.0, 2.0, 10))
	subplt.Title("neural network predictions")
	subplt.XLabel("t values")
	subplt.YLabel("x1")
	subplt.Legend("analytical model", "model prediction", "end of training window")
	subplt.XLim(0.0, 40.0)
	subplt.Grid()

	subplt = plt.Subplot(1, 2)
	subplt.Grid()
	subplt.Plot(times, mat.Col(nil, 1, data))
	subplt.Plot(times, mat.Col(nil, 1, integratedPred))
	subplt.Plot(ngo.Linspace(times[trainingDim], times[trainingDim], 10), ngo.Linspace(-2.0, 2.0, 10))
	subplt.Title("neural network predictions")
	subplt.XLabel("t values")
	subplt.YLabel("x2")
	subplt.Legend("analytical model", "model prediction", "end of training window")
	subplt.XLim(0.0, 40.0)

	plt.Save("plot.png")
	plt.Show()
}
