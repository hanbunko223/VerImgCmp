package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
)

func TestSplitDegreeExponents(t *testing.T) {
	cases := map[int][4]int{
		19: {5, 5, 5, 4},
		20: {5, 5, 5, 5},
		21: {6, 5, 5, 5},
		22: {6, 6, 5, 5},
		23: {6, 6, 6, 5},
	}
	for input, want := range cases {
		got := [4]int{}
		got[0], got[1], got[2], got[3] = splitDegreeExponents(input)
		if got != want {
			t.Fatalf("splitDegreeExponents(%d) = %v, want %v", input, got, want)
		}
	}
}

func TestSetupCommitOpenVerifySmall(t *testing.T) {
	for _, numVars := range []int{4, 5, 7} {
		srs, err := setup(numVars)
		if err != nil {
			t.Fatalf("setup(%d): %v", numVars, err)
		}
		poly := randomPolynomial(numVars)
		point := randomPoint(numVars)
		value := poly.evaluate(point)
		commitment, aux, err := commit(srs, poly)
		if err != nil {
			t.Fatalf("commit(%d): %v", numVars, err)
		}
		opening, err := open(srs, point, aux, poly)
		if err != nil {
			t.Fatalf("open(%d): %v", numVars, err)
		}
		if err := verify(srs, point, &value, commitment, opening); err != nil {
			t.Fatalf("verify(%d): %v", numVars, err)
		}
	}
}

func TestUnevenSplitRegression(t *testing.T) {
	srs, err := setup(5)
	if err != nil {
		t.Fatal(err)
	}
	if srs.DegreeX != 4 || srs.DegreeY != 2 || srs.DegreeZ != 2 || srs.DegreeT != 2 {
		t.Fatalf("unexpected uneven split: %+v", srs)
	}

	poly := randomPolynomial(5)
	point := randomPoint(5)
	value := poly.evaluate(point)
	commitment, aux, err := commit(srs, poly)
	if err != nil {
		t.Fatal(err)
	}
	opening, err := open(srs, point, aux, poly)
	if err != nil {
		t.Fatal(err)
	}
	if err := verify(srs, point, &value, commitment, opening); err != nil {
		t.Fatal(err)
	}
}

func TestNegativeClaimFails(t *testing.T) {
	srs, err := setup(5)
	if err != nil {
		t.Fatal(err)
	}
	poly := randomPolynomial(5)
	point := randomPoint(5)
	value := poly.evaluate(point)
	commitment, aux, err := commit(srs, poly)
	if err != nil {
		t.Fatal(err)
	}
	opening, err := open(srs, point, aux, poly)
	if err != nil {
		t.Fatal(err)
	}
	var badValue fr.Element
	badValue.Set(&value)
	var one fr.Element
	one.SetOne()
	badValue.Add(&badValue, &one)
	if err := verify(srs, point, &badValue, commitment, opening); err == nil {
		t.Fatal("expected verification to fail for mutated claimed value")
	}
}

func TestRustCrossCheckIfAvailable(t *testing.T) {
	rustBin := filepath.Clean("../kzh_fold/target/release/zkcnn_kzh_cli")
	if _, err := os.Stat(rustBin); err != nil {
		t.Skip("rust KZH4 sidecar not built; skipping cross-check")
	}

	numVars := 5
	srs, err := setup(numVars)
	if err != nil {
		t.Fatal(err)
	}
	poly := randomPolynomial(numVars)
	pointCxx := randomPoint(numVars)
	pointKzh := append([]fr.Element(nil), pointCxx...)
	reverseScalars(pointKzh)
	value := poly.evaluate(pointKzh)
	commitment, aux, err := commit(srs, poly)
	if err != nil {
		t.Fatal(err)
	}
	opening, err := open(srs, pointKzh, aux, poly)
	if err != nil {
		t.Fatal(err)
	}
	if err := verify(srs, pointKzh, &value, commitment, opening); err != nil {
		t.Fatal(err)
	}

	tempDir := t.TempDir()
	goSrs := filepath.Join(tempDir, "go.srs")
	goArtifact := filepath.Join(tempDir, "go.artifact")
	goVerifyMetrics := filepath.Join(tempDir, "go.verify.metrics")
	rustSrs := filepath.Join(tempDir, "rust.srs")
	rustArtifact := filepath.Join(tempDir, "rust.artifact")
	rustProveMetrics := filepath.Join(tempDir, "rust.prove.metrics")
	rustVerifyMetrics := filepath.Join(tempDir, "rust.verify.metrics")
	polyPath := filepath.Join(tempDir, "poly.bin")
	pointPath := filepath.Join(tempDir, "point.bin")
	valuePath := filepath.Join(tempDir, "value.bin")
	badValuePath := filepath.Join(tempDir, "bad_value.bin")

	if err := writeSRS(goSrs, srs); err != nil {
		t.Fatal(err)
	}
	if err := writeArtifact(goArtifact, commitment, opening); err != nil {
		t.Fatal(err)
	}
	if err := writeCxxScalarVector(polyPath, poly.evaluations); err != nil {
		t.Fatal(err)
	}
	if err := writeCxxScalarVector(pointPath, pointCxx); err != nil {
		t.Fatal(err)
	}
	if err := writeCxxScalarValue(valuePath, value); err != nil {
		t.Fatal(err)
	}
	var badValue fr.Element
	badValue.Set(&value)
	var one fr.Element
	one.SetOne()
	badValue.Add(&badValue, &one)
	if err := writeCxxScalarValue(badValuePath, badValue); err != nil {
		t.Fatal(err)
	}

	if err := runCmd(exec.Command(rustBin, "setup", "--num-vars", strconv.Itoa(numVars), "--srs", rustSrs)); err != nil {
		t.Skipf("rust KZH4 helper not runnable; skipping cross-check: %v", err)
	}
	if err := runCmd(exec.Command(rustBin, "prove", "--srs", rustSrs, "--poly", polyPath, "--point", pointPath, "--artifact", rustArtifact, "--metrics", rustProveMetrics)); err != nil {
		t.Fatalf("rust prove failed: %v", err)
	}
	if err := runCmd(exec.Command(rustBin, "verify", "--srs", rustSrs, "--point", pointPath, "--value", valuePath, "--artifact", rustArtifact, "--metrics", rustVerifyMetrics)); err != nil {
		t.Fatalf("rust verify failed: %v", err)
	}
	if err := runCmd(exec.Command(rustBin, "verify", "--srs", rustSrs, "--point", pointPath, "--value", badValuePath, "--artifact", rustArtifact, "--metrics", rustVerifyMetrics)); err == nil {
		t.Fatal("rust sidecar unexpectedly accepted a tampered claim")
	}
	decodedCommitment, decodedOpening, err := readArtifact(goArtifact)
	if err != nil {
		t.Fatal(err)
	}
	decodedSRS, err := readSRS(goSrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := verify(decodedSRS, pointKzh, &value, decodedCommitment, decodedOpening); err != nil {
		t.Fatalf("go verify failed: %v", err)
	}
	if err := verify(decodedSRS, pointKzh, &badValue, decodedCommitment, decodedOpening); err == nil {
		t.Fatal("go sidecar unexpectedly accepted a tampered claim")
	}
	_ = goVerifyMetrics
}

func randomPolynomial(numVars int) multilinearPolynomial {
	evals := make([]fr.Element, 1<<numVars)
	for i := range evals {
		_, err := evals[i].SetRandom()
		if err != nil {
			panic(err)
		}
	}
	return newMultilinearPolynomial(evals)
}

func randomPoint(numVars int) []fr.Element {
	out := make([]fr.Element, numVars)
	for i := range out {
		_, err := out[i].SetRandom()
		if err != nil {
			panic(err)
		}
	}
	return out
}

func runCmd(cmd *exec.Cmd) error {
	cmd.Env = os.Environ()
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, string(output))
	}
	return nil
}
