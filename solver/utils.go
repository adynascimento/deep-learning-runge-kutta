package solver

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func LoadFromFile(path string) *mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading features from file:", err.Error())
	}
	defer file.Close()

	lines, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Println("error reading features from file:", err.Error())
	}

	m := mat.NewDense(len(lines[0]), len(lines), nil)
	for j, line := range lines {
		for i, col := range line {
			value, _ := strconv.ParseFloat(col, 64)
			m.Set(i, j, value)
		}
	}

	return m
}

// split dataset into two matrix (training and testing)
func Split(a *mat.Dense, frac float64) (*mat.Dense, *mat.Dense) {
	nRows, nCols := a.Dims()

	jdx := int(frac * float64(nCols))
	m1 := a.Slice(0, nRows, 0, jdx)
	m2 := a.Slice(0, nRows, jdx, nCols)

	return m1.(*mat.Dense), m2.(*mat.Dense)
}
