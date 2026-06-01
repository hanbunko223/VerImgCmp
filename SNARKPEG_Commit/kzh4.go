package main

import (
	"errors"
	"fmt"
	"math/big"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc"
	bls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
)

type multilinearPolynomial struct {
	numVariables int
	evaluations  []fr.Element
}

type kzh4SRS struct {
	DegreeX int
	DegreeY int
	DegreeZ int
	DegreeT int

	HXYZT []bls12381.G1Affine
	HYZT  []bls12381.G1Affine
	HZT   []bls12381.G1Affine
	HT    []bls12381.G1Affine

	VX []bls12381.G2Affine
	VY []bls12381.G2Affine
	VZ []bls12381.G2Affine
	VT []bls12381.G2Affine

	V bls12381.G2Affine
}

type kzh4Commitment struct {
	C bls12381.G1Affine
}

type kzh4Aux struct {
	DX  []bls12381.G1Affine
	DXY []bls12381.G1Affine
}

type kzh4Opening struct {
	DX    []bls12381.G1Affine
	DY    []bls12381.G1Affine
	DZ    []bls12381.G1Affine
	FStar multilinearPolynomial
}

func newMultilinearPolynomial(evaluations []fr.Element) multilinearPolynomial {
	copied := append([]fr.Element(nil), evaluations...)
	return multilinearPolynomial{
		numVariables: log2Exact(len(copied)),
		evaluations:  copied,
	}
}

func (p multilinearPolynomial) clone() multilinearPolynomial {
	return newMultilinearPolynomial(p.evaluations)
}

func (p multilinearPolynomial) len() int {
	return len(p.evaluations)
}

func (p multilinearPolynomial) extendNumberOfVariables(numVariables int) multilinearPolynomial {
	if p.numVariables > numVariables {
		panic("cannot shrink multilinear polynomial")
	}
	if p.numVariables == numVariables {
		return p.clone()
	}
	targetLen := 1 << numVariables
	out := make([]fr.Element, targetLen)
	copy(out, p.evaluations)
	return multilinearPolynomial{
		numVariables: numVariables,
		evaluations:  out,
	}
}

func (p *multilinearPolynomial) boundPolyVarTop(r *fr.Element) {
	n := len(p.evaluations) / 2
	for i := 0; i < n; i++ {
		var delta fr.Element
		delta.Sub(&p.evaluations[i+n], &p.evaluations[i])
		delta.Mul(&delta, r)
		p.evaluations[i].Add(&p.evaluations[i], &delta)
	}
	p.numVariables--
	p.evaluations = append([]fr.Element(nil), p.evaluations[:n]...)
}

func (p multilinearPolynomial) partialEvaluation(fixedVars []fr.Element) multilinearPolynomial {
	tmp := p.clone()
	for i := range fixedVars {
		tmp.boundPolyVarTop(&fixedVars[i])
	}
	return tmp
}

func (p multilinearPolynomial) getPartialEvaluationForBooleanInput(index int, width int) []fr.Element {
	start := width * index
	end := start + width
	return append([]fr.Element(nil), p.evaluations[start:end]...)
}

func (p multilinearPolynomial) evaluate(point []fr.Element) fr.Element {
	if len(point) != p.numVariables {
		panic("mismatched point dimension")
	}
	eqs := eqEvals(point)
	return innerProduct(p.evaluations, eqs)
}

func log2Exact(n int) int {
	if n <= 0 || n&(n-1) != 0 {
		panic(fmt.Sprintf("value is not a power of two: %d", n))
	}
	return bits.Len(uint(n)) - 1
}

func splitDegreeExponents(n int) (int, int, int, int) {
	switch n % 4 {
	case 0:
		return n / 4, n / 4, n / 4, n / 4
	case 1:
		return n/4 + 1, n / 4, n / 4, n / 4
	case 2:
		return n/4 + 1, n/4 + 1, n / 4, n / 4
	case 3:
		return n/4 + 1, n/4 + 1, n/4 + 1, n / 4
	default:
		panic("unreachable")
	}
}

func splitInput(srs *kzh4SRS, input []fr.Element) ([][]fr.Element, error) {
	totalLength := log2Exact(srs.DegreeX) + log2Exact(srs.DegreeY) + log2Exact(srs.DegreeZ) + log2Exact(srs.DegreeT)
	extended := append([]fr.Element(nil), input...)
	if len(extended) < totalLength {
		padded := make([]fr.Element, totalLength-len(extended))
		extended = append(padded, extended...)
	}
	if len(extended) != totalLength {
		return nil, fmt.Errorf("input length mismatch: got %d, want %d", len(extended), totalLength)
	}

	xBits := log2Exact(srs.DegreeX)
	yBits := log2Exact(srs.DegreeY)
	zBits := log2Exact(srs.DegreeZ)
	return [][]fr.Element{
		append([]fr.Element(nil), extended[:xBits]...),
		append([]fr.Element(nil), extended[xBits:xBits+yBits]...),
		append([]fr.Element(nil), extended[xBits+yBits:xBits+yBits+zBits]...),
		append([]fr.Element(nil), extended[xBits+yBits+zBits:]...),
	}, nil
}

func eqEvals(r []fr.Element) []fr.Element {
	evals := make([]fr.Element, 1<<len(r))
	var one fr.Element
	one.SetOne()
	evals[0] = one
	size := 1
	for j := 0; j < len(r); j++ {
		for i := size - 1; i >= 0; i-- {
			scalar := evals[i]
			var hi fr.Element
			hi.Mul(&scalar, &r[j])
			var lo fr.Element
			lo.Sub(&scalar, &hi)
			evals[2*i] = lo
			evals[2*i+1] = hi
		}
		size *= 2
	}
	return evals[:size]
}

func innerProduct(left, right []fr.Element) fr.Element {
	if len(left) != len(right) {
		panic("inner product length mismatch")
	}
	var acc fr.Element
	for i := range left {
		var term fr.Element
		term.Mul(&left[i], &right[i])
		acc.Add(&acc, &term)
	}
	return acc
}

func randScalar() (fr.Element, error) {
	var out fr.Element
	_, err := out.SetRandom()
	return out, err
}

func scalarToBigInt(x *fr.Element) *big.Int {
	return x.BigInt(new(big.Int))
}

func sampleRandomG1(base *bls12381.G1Affine) (bls12381.G1Affine, error) {
	r, err := randScalar()
	if err != nil {
		return bls12381.G1Affine{}, err
	}
	var out bls12381.G1Affine
	out.ScalarMultiplication(base, scalarToBigInt(&r))
	return out, nil
}

func sampleRandomG2(base *bls12381.G2Affine) (bls12381.G2Affine, error) {
	r, err := randScalar()
	if err != nil {
		return bls12381.G2Affine{}, err
	}
	var out bls12381.G2Affine
	out.ScalarMultiplication(base, scalarToBigInt(&r))
	return out, nil
}

func setup(maximumDegree int) (*kzh4SRS, error) {
	expX, expY, expZ, expT := splitDegreeExponents(maximumDegree)
	degreeX := 1 << expX
	degreeY := 1 << expY
	degreeZ := 1 << expZ
	degreeT := 1 << expT

	_, _, baseG1, baseG2 := bls12381.Generators()
	g, err := sampleRandomG1(&baseG1)
	if err != nil {
		return nil, err
	}
	v, err := sampleRandomG2(&baseG2)
	if err != nil {
		return nil, err
	}

	tauX, err := randomScalarSlice(degreeX)
	if err != nil {
		return nil, err
	}
	tauY, err := randomScalarSlice(degreeY)
	if err != nil {
		return nil, err
	}
	tauZ, err := randomScalarSlice(degreeZ)
	if err != nil {
		return nil, err
	}
	tauT, err := randomScalarSlice(degreeT)
	if err != nil {
		return nil, err
	}

	tauZT := make([]fr.Element, degreeZ*degreeT)
	for z := 0; z < degreeZ; z++ {
		for t := 0; t < degreeT; t++ {
			idx := z*degreeT + t
			tauZT[idx].Mul(&tauZ[z], &tauT[t])
		}
	}

	tauYZT := make([]fr.Element, degreeY*degreeZ*degreeT)
	for y := 0; y < degreeY; y++ {
		offset := y * degreeZ * degreeT
		for idx := 0; idx < len(tauZT); idx++ {
			tauYZT[offset+idx].Mul(&tauY[y], &tauZT[idx])
		}
	}

	tauXYZT := make([]fr.Element, degreeX*degreeY*degreeZ*degreeT)
	blockSize := degreeY * degreeZ * degreeT
	for x := 0; x < degreeX; x++ {
		offset := x * blockSize
		for idx := 0; idx < blockSize; idx++ {
			tauXYZT[offset+idx].Mul(&tauX[x], &tauYZT[idx])
		}
	}

	hXYZT := bls12381.BatchScalarMultiplicationG1(&g, tauXYZT)
	hYZT := bls12381.BatchScalarMultiplicationG1(&g, tauYZT)
	hZT := bls12381.BatchScalarMultiplicationG1(&g, tauZT)
	hT := bls12381.BatchScalarMultiplicationG1(&g, tauT)
	vX := bls12381.BatchScalarMultiplicationG2(&v, tauX)
	vY := bls12381.BatchScalarMultiplicationG2(&v, tauY)
	vZ := bls12381.BatchScalarMultiplicationG2(&v, tauZ)
	vT := bls12381.BatchScalarMultiplicationG2(&v, tauT)

	return &kzh4SRS{
		DegreeX: degreeX,
		DegreeY: degreeY,
		DegreeZ: degreeZ,
		DegreeT: degreeT,
		HXYZT:   hXYZT,
		HYZT:    hYZT,
		HZT:     hZT,
		HT:      hT,
		VX:      vX,
		VY:      vY,
		VZ:      vZ,
		VT:      vT,
		V:       v,
	}, nil
}

func randomScalarSlice(n int) ([]fr.Element, error) {
	out := make([]fr.Element, n)
	for i := 0; i < n; i++ {
		_, err := out[i].SetRandom()
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func commit(srs *kzh4SRS, poly multilinearPolynomial) (kzh4Commitment, kzh4Aux, error) {
	totalVars := log2Exact(srs.DegreeX) + log2Exact(srs.DegreeY) + log2Exact(srs.DegreeZ) + log2Exact(srs.DegreeT)
	extended := poly.extendNumberOfVariables(totalVars)
	if extended.len() != srs.DegreeX*srs.DegreeY*srs.DegreeZ*srs.DegreeT {
		return kzh4Commitment{}, kzh4Aux{}, errors.New("extended polynomial length mismatch")
	}

	c, err := msmG1(srs.HXYZT, extended.evaluations)
	if err != nil {
		return kzh4Commitment{}, kzh4Aux{}, err
	}

	dx := make([]bls12381.G1Affine, srs.DegreeX)
	sliceWidthX := srs.DegreeY * srs.DegreeZ * srs.DegreeT
	for i := 0; i < srs.DegreeX; i++ {
		evals := extended.getPartialEvaluationForBooleanInput(i, sliceWidthX)
		dx[i], err = msmG1(srs.HYZT, evals)
		if err != nil {
			return kzh4Commitment{}, kzh4Aux{}, err
		}
	}

	dxy := make([]bls12381.G1Affine, srs.DegreeX*srs.DegreeY)
	sliceWidthXY := srs.DegreeZ * srs.DegreeT
	for i := 0; i < len(dxy); i++ {
		evals := extended.getPartialEvaluationForBooleanInput(i, sliceWidthXY)
		dxy[i], err = msmG1(srs.HZT, evals)
		if err != nil {
			return kzh4Commitment{}, kzh4Aux{}, err
		}
	}

	return kzh4Commitment{C: c}, kzh4Aux{DX: dx, DXY: dxy}, nil
}

func open(srs *kzh4SRS, input []fr.Element, aux kzh4Aux, poly multilinearPolynomial) (kzh4Opening, error) {
	totalVars := log2Exact(srs.DegreeX) + log2Exact(srs.DegreeY) + log2Exact(srs.DegreeZ) + log2Exact(srs.DegreeT)
	extended := poly.extendNumberOfVariables(totalVars)
	split, err := splitInput(srs, input)
	if err != nil {
		return kzh4Opening{}, err
	}

	eqX := eqEvals(split[0])
	groupedDXY := make([][]bls12381.G1Affine, srs.DegreeY)
	for j, val := range aux.DXY {
		i := j % srs.DegreeY
		groupedDXY[i] = append(groupedDXY[i], val)
	}
	for i := range groupedDXY {
		if len(groupedDXY[i]) != srs.DegreeX {
			return kzh4Opening{}, fmt.Errorf("wrong grouped DXY width for y=%d: got %d want %d", i, len(groupedDXY[i]), srs.DegreeX)
		}
	}

	dy := make([]bls12381.G1Affine, srs.DegreeY)
	for i := range groupedDXY {
		dy[i], err = msmG1(groupedDXY[i], eqX)
		if err != nil {
			return kzh4Opening{}, err
		}
	}

	xyInput := append(append([]fr.Element(nil), split[0]...), split[1]...)
	partialXY := extended.partialEvaluation(xyInput)

	dz := make([]bls12381.G1Affine, srs.DegreeZ)
	for i := 0; i < srs.DegreeZ; i++ {
		evals := partialXY.getPartialEvaluationForBooleanInput(i, srs.DegreeT)
		dz[i], err = msmG1(srs.HT, evals)
		if err != nil {
			return kzh4Opening{}, err
		}
	}

	xyzInput := append(xyInput, split[2]...)
	fStar := extended.partialEvaluation(xyzInput)
	return kzh4Opening{
		DX:    append([]bls12381.G1Affine(nil), aux.DX...),
		DY:    dy,
		DZ:    dz,
		FStar: fStar,
	}, nil
}

func verify(srs *kzh4SRS, input []fr.Element, output *fr.Element, commitment kzh4Commitment, opening kzh4Opening) error {
	split, err := splitInput(srs, input)
	if err != nil {
		return err
	}

	lhs, err := bls12381.Pair(opening.DX, srs.VX)
	if err != nil {
		return err
	}
	rhs, err := bls12381.Pair([]bls12381.G1Affine{commitment.C}, []bls12381.G2Affine{srs.V})
	if err != nil {
		return err
	}
	if !lhs.Equal(&rhs) {
		return errors.New("D_x pairing check failed")
	}

	newC, err := msmG1(opening.DX, eqEvals(split[0]))
	if err != nil {
		return err
	}
	lhs, err = bls12381.Pair(opening.DY, srs.VY)
	if err != nil {
		return err
	}
	rhs, err = bls12381.Pair([]bls12381.G1Affine{newC}, []bls12381.G2Affine{srs.V})
	if err != nil {
		return err
	}
	if !lhs.Equal(&rhs) {
		return errors.New("D_y pairing check failed")
	}

	newC, err = msmG1(opening.DY, eqEvals(split[1]))
	if err != nil {
		return err
	}
	lhs, err = bls12381.Pair(opening.DZ, srs.VZ)
	if err != nil {
		return err
	}
	rhs, err = bls12381.Pair([]bls12381.G1Affine{newC}, []bls12381.G2Affine{srs.V})
	if err != nil {
		return err
	}
	if !lhs.Equal(&rhs) {
		return errors.New("D_z pairing check failed")
	}

	lhsPoint, err := msmG1(srs.HT, opening.FStar.evaluations)
	if err != nil {
		return err
	}
	rhsPoint, err := msmG1(opening.DZ, eqEvals(split[2]))
	if err != nil {
		return err
	}
	if !lhsPoint.Equal(&rhsPoint) {
		return errors.New("f_star commitment check failed")
	}

	got := opening.FStar.evaluate(split[3])
	if !got.Equal(output) {
		return errors.New("f_star evaluation mismatch")
	}
	return nil
}

func msmG1(points []bls12381.G1Affine, scalars []fr.Element) (bls12381.G1Affine, error) {
	if len(points) != len(scalars) {
		return bls12381.G1Affine{}, fmt.Errorf("G1 MSM length mismatch: %d vs %d", len(points), len(scalars))
	}
	var out bls12381.G1Affine
	_, err := out.MultiExp(points, scalars, ecc.MultiExpConfig{})
	return out, err
}
