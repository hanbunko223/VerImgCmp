package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	bls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
)

const (
	srsMagic      = "KZH4GNS1"
	artifactMagic = "KZH4GNA1"
)

func readScalarVectorCxx(path string) ([]fr.Element, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	length, err := readU64(reader)
	if err != nil {
		return nil, err
	}
	values := make([]fr.Element, int(length))
	for i := range values {
		elem, err := readFrLittleEndian(reader)
		if err != nil {
			return nil, err
		}
		values[i] = elem
	}
	return values, nil
}

func readScalarValueCxx(path string) (fr.Element, error) {
	file, err := os.Open(path)
	if err != nil {
		return fr.Element{}, err
	}
	defer file.Close()
	return readFrLittleEndian(bufio.NewReader(file))
}

func writeCxxScalarVector(path string, values []fr.Element) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	if err := writeU64(writer, uint64(len(values))); err != nil {
		return err
	}
	for i := range values {
		var buf [fr.Bytes]byte
		fr.LittleEndian.PutElement(&buf, values[i])
		if _, err := writer.Write(buf[:]); err != nil {
			return err
		}
	}
	return writer.Flush()
}

func writeCxxScalarValue(path string, value fr.Element) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	var buf [fr.Bytes]byte
	fr.LittleEndian.PutElement(&buf, value)
	if _, err := writer.Write(buf[:]); err != nil {
		return err
	}
	return writer.Flush()
}

func writeMetrics(path string, values map[string]string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for key, value := range values {
		if _, err := fmt.Fprintf(writer, "%s %s\n", key, value); err != nil {
			return err
		}
	}
	return writer.Flush()
}

func writeSRS(path string, srs *kzh4SRS) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	if _, err := writer.WriteString(srsMagic); err != nil {
		return err
	}
	for _, degree := range []int{srs.DegreeX, srs.DegreeY, srs.DegreeZ, srs.DegreeT} {
		if err := writeU64(writer, uint64(degree)); err != nil {
			return err
		}
	}

	if err := writeG1Slice(writer, srs.HXYZT); err != nil {
		return err
	}
	if err := writeG1Slice(writer, srs.HYZT); err != nil {
		return err
	}
	if err := writeG1Slice(writer, srs.HZT); err != nil {
		return err
	}
	if err := writeG1Slice(writer, srs.HT); err != nil {
		return err
	}
	if err := writeG2Slice(writer, srs.VX); err != nil {
		return err
	}
	if err := writeG2Slice(writer, srs.VY); err != nil {
		return err
	}
	if err := writeG2Slice(writer, srs.VZ); err != nil {
		return err
	}
	if err := writeG2Slice(writer, srs.VT); err != nil {
		return err
	}
	if err := writeG2(writer, srs.V); err != nil {
		return err
	}
	return writer.Flush()
}

func readSRS(path string) (*kzh4SRS, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	magic := make([]byte, len(srsMagic))
	if _, err := io.ReadFull(reader, magic); err != nil {
		return nil, err
	}
	if string(magic) != srsMagic {
		return nil, fmt.Errorf("invalid SRS magic: %q", string(magic))
	}

	degreeX, err := readU64(reader)
	if err != nil {
		return nil, err
	}
	degreeY, err := readU64(reader)
	if err != nil {
		return nil, err
	}
	degreeZ, err := readU64(reader)
	if err != nil {
		return nil, err
	}
	degreeT, err := readU64(reader)
	if err != nil {
		return nil, err
	}

	hXYZT, err := readG1Slice(reader)
	if err != nil {
		return nil, err
	}
	hYZT, err := readG1Slice(reader)
	if err != nil {
		return nil, err
	}
	hZT, err := readG1Slice(reader)
	if err != nil {
		return nil, err
	}
	hT, err := readG1Slice(reader)
	if err != nil {
		return nil, err
	}
	vX, err := readG2Slice(reader)
	if err != nil {
		return nil, err
	}
	vY, err := readG2Slice(reader)
	if err != nil {
		return nil, err
	}
	vZ, err := readG2Slice(reader)
	if err != nil {
		return nil, err
	}
	vT, err := readG2Slice(reader)
	if err != nil {
		return nil, err
	}
	v, err := readG2(reader)
	if err != nil {
		return nil, err
	}

	return &kzh4SRS{
		DegreeX: int(degreeX),
		DegreeY: int(degreeY),
		DegreeZ: int(degreeZ),
		DegreeT: int(degreeT),
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

func writeArtifact(path string, commitment kzh4Commitment, opening kzh4Opening) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	if _, err := writer.WriteString(artifactMagic); err != nil {
		return err
	}
	if err := writeG1(writer, commitment.C); err != nil {
		return err
	}
	if err := writeG1Slice(writer, opening.DX); err != nil {
		return err
	}
	if err := writeG1Slice(writer, opening.DY); err != nil {
		return err
	}
	if err := writeG1Slice(writer, opening.DZ); err != nil {
		return err
	}
	if err := writeU64(writer, uint64(opening.FStar.numVariables)); err != nil {
		return err
	}
	if err := writeFrBigEndianSlice(writer, opening.FStar.evaluations); err != nil {
		return err
	}
	return writer.Flush()
}

func readArtifact(path string) (kzh4Commitment, kzh4Opening, error) {
	file, err := os.Open(path)
	if err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	magic := make([]byte, len(artifactMagic))
	if _, err := io.ReadFull(reader, magic); err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}
	if string(magic) != artifactMagic {
		return kzh4Commitment{}, kzh4Opening{}, fmt.Errorf("invalid artifact magic: %q", string(magic))
	}

	commitmentPoint, err := readG1(reader)
	if err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}
	dx, err := readG1Slice(reader)
	if err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}
	dy, err := readG1Slice(reader)
	if err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}
	dz, err := readG1Slice(reader)
	if err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}
	numVars, err := readU64(reader)
	if err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}
	evals, err := readFrBigEndianSlice(reader)
	if err != nil {
		return kzh4Commitment{}, kzh4Opening{}, err
	}

	return kzh4Commitment{C: commitmentPoint}, kzh4Opening{
		DX: dx,
		DY: dy,
		DZ: dz,
		FStar: multilinearPolynomial{
			numVariables: int(numVars),
			evaluations:  evals,
		},
	}, nil
}

func proofSizeBytes(commitment kzh4Commitment, opening kzh4Opening) (uint64, uint64) {
	commitmentSize := uint64(len(commitment.C.Bytes()))
	openingSize := uint64(0)
	var zeroG1 bls12381.G1Affine
	g1Size := uint64(len(zeroG1.Bytes()))
	for range opening.DX {
		openingSize += g1Size
	}
	for range opening.DY {
		openingSize += g1Size
	}
	for range opening.DZ {
		openingSize += g1Size
	}
	for range opening.FStar.evaluations {
		openingSize += uint64(fr.Bytes)
	}
	return commitmentSize, openingSize
}

func writeU64(writer io.Writer, value uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], value)
	_, err := writer.Write(buf[:])
	return err
}

func readU64(reader io.Reader) (uint64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(reader, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

func readFrLittleEndian(reader io.Reader) (fr.Element, error) {
	var buf [fr.Bytes]byte
	if _, err := io.ReadFull(reader, buf[:]); err != nil {
		return fr.Element{}, err
	}
	return fr.LittleEndian.Element(&buf)
}

func writeFrBigEndianSlice(writer io.Writer, values []fr.Element) error {
	if err := writeU64(writer, uint64(len(values))); err != nil {
		return err
	}
	for i := range values {
		var buf [fr.Bytes]byte
		fr.BigEndian.PutElement(&buf, values[i])
		if _, err := writer.Write(buf[:]); err != nil {
			return err
		}
	}
	return nil
}

func readFrBigEndianSlice(reader io.Reader) ([]fr.Element, error) {
	length, err := readU64(reader)
	if err != nil {
		return nil, err
	}
	values := make([]fr.Element, int(length))
	for i := range values {
		var buf [fr.Bytes]byte
		if _, err := io.ReadFull(reader, buf[:]); err != nil {
			return nil, err
		}
		values[i], err = fr.BigEndian.Element(&buf)
		if err != nil {
			return nil, err
		}
	}
	return values, nil
}

func writeG1(writer io.Writer, point bls12381.G1Affine) error {
	buf := point.Bytes()
	_, err := writer.Write(buf[:])
	return err
}

func readG1(reader io.Reader) (bls12381.G1Affine, error) {
	var point bls12381.G1Affine
	buf := make([]byte, bls12381.SizeOfG1AffineCompressed)
	if _, err := io.ReadFull(reader, buf); err != nil {
		return point, err
	}
	_, err := point.SetBytes(buf)
	return point, err
}

func writeG2(writer io.Writer, point bls12381.G2Affine) error {
	buf := point.Bytes()
	_, err := writer.Write(buf[:])
	return err
}

func readG2(reader io.Reader) (bls12381.G2Affine, error) {
	var point bls12381.G2Affine
	buf := make([]byte, bls12381.SizeOfG2AffineCompressed)
	if _, err := io.ReadFull(reader, buf); err != nil {
		return point, err
	}
	_, err := point.SetBytes(buf)
	return point, err
}

func writeG1Slice(writer io.Writer, points []bls12381.G1Affine) error {
	if err := writeU64(writer, uint64(len(points))); err != nil {
		return err
	}
	for i := range points {
		if err := writeG1(writer, points[i]); err != nil {
			return err
		}
	}
	return nil
}

func readG1Slice(reader io.Reader) ([]bls12381.G1Affine, error) {
	length, err := readU64(reader)
	if err != nil {
		return nil, err
	}
	points := make([]bls12381.G1Affine, int(length))
	for i := range points {
		points[i], err = readG1(reader)
		if err != nil {
			return nil, err
		}
	}
	return points, nil
}

func writeG2Slice(writer io.Writer, points []bls12381.G2Affine) error {
	if err := writeU64(writer, uint64(len(points))); err != nil {
		return err
	}
	for i := range points {
		if err := writeG2(writer, points[i]); err != nil {
			return err
		}
	}
	return nil
}

func readG2Slice(reader io.Reader) ([]bls12381.G2Affine, error) {
	length, err := readU64(reader)
	if err != nil {
		return nil, err
	}
	points := make([]bls12381.G2Affine, int(length))
	for i := range points {
		points[i], err = readG2(reader)
		if err != nil {
			return nil, err
		}
	}
	return points, nil
}
