package solver

import (
	ngo "github.com/adynascimento/deep-learning/gonum"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"gonum.org/v1/gonum/mat"
)

func SolveRK4(model network.NeuralModel, x0 []float64, tmax int, h float64) *mat.Dense {
	var sumK *mat.Dense
	x := mat.NewDense(len(x0), 1, nil)
	s := mat.NewDense(len(x0), tmax, nil)

	// initial solution
	s.SetCol(0, x0)
	for j := 0; j < tmax-1; j++ {
		x.SetCol(0, mat.Col(nil, j, s))
		k1 := model.Predict(x)
		k2 := model.Predict(ngo.Add(x, ngo.Scale(0.5*h, k1)))
		k3 := model.Predict(ngo.Add(x, ngo.Scale(0.5*h, k2)))
		k4 := model.Predict(ngo.Add(x, ngo.Scale(h, k3)))

		sumK = ngo.Add(k1, ngo.Scale(2.0, k2))
		sumK = ngo.Add(sumK, ngo.Scale(2.0, k3))
		sumK = ngo.Add(sumK, k4)

		s.SetCol(j+1, ngo.Add(x, ngo.Scale(h/6.0, sumK)).RawMatrix().Data)
	}

	return s
}
