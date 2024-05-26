package solver

import (
	ngo "github.com/adynascimento/deep-learning/gonum"
	"gonum.org/v1/gonum/mat"
)

type Parameters struct {
	Func func(x *mat.Dense) *mat.Dense
	X0   []float64
	Tmax float64
	Step float64
}

func SolveRK4(p Parameters) *mat.Dense {
	steps := int(p.Tmax/p.Step) + 1

	var sumK *mat.Dense
	x := mat.NewDense(len(p.X0), 1, nil)
	s := mat.NewDense(len(p.X0), steps, nil)

	// initial solution
	s.SetCol(0, p.X0)
	for j := 1; j < steps; j++ {
		x.SetCol(0, mat.Col(nil, j-1, s))
		k1 := p.Func(x)
		k2 := p.Func(ngo.Add(x, ngo.Scale(0.5*p.Step, k1)))
		k3 := p.Func(ngo.Add(x, ngo.Scale(0.5*p.Step, k2)))
		k4 := p.Func(ngo.Add(x, ngo.Scale(p.Step, k3)))

		sumK = ngo.Add(k1, ngo.Scale(2.0, k2))
		sumK = ngo.Add(sumK, ngo.Scale(2.0, k3))
		sumK = ngo.Add(sumK, k4)

		s.SetCol(j, ngo.Add(x, ngo.Scale(p.Step/6.0, sumK)).RawMatrix().Data)
	}

	return s
}
