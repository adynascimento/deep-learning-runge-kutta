package numeric

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func readFromFile(path string) [][]string {
	file, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		fmt.Println(err)
	}

	return lines
}

func ReadData(path string) ([]float64, [][]float64) {
	data := readFromFile(path)

	x := []float64{}
	y := make([][]float64, 2, len(data))
	for _, line := range data {
		time, _ := strconv.ParseFloat(strings.Fields(line[0])[0], 64)
		feature_1, _ := strconv.ParseFloat(strings.Fields(line[0])[1], 64)
		feature_2, _ := strconv.ParseFloat(strings.Fields(line[0])[2], 64)

		x = append(x, time)
		y[0] = append(y[0], feature_1)
		y[1] = append(y[1], feature_2)
	}
	
	return x, y
}

func ReadDerivativeData(path string) [][]float64 {
	data := readFromFile(path)

	y := make([][]float64, 2, len(data))
	for _, line := range data {
		feature_1, _ := strconv.ParseFloat(strings.Fields(line[0])[0], 64)
		feature_2, _ := strconv.ParseFloat(strings.Fields(line[0])[1], 64)

		y[0] = append(y[0], feature_1)
		y[1] = append(y[1], feature_2)
	}

	return y
}
