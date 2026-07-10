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

	if err := writeG1SliceRaw(writer, srs.HXYZT); err != nil {
		return err
	}
	if err := writeG1SliceRaw(writer, srs.HYZT); err != nil {
		return err
	}
	if err := writeG1SliceRaw(writer, srs.HZT); err != nil {
		return err
	}
	if err := writeG1SliceRaw(writer, srs.HT); err != nil {
		return err
	}
	if err := writeG2SliceRaw(writer, srs.VX); err != nil {
		return err
	}
	if err := writeG2SliceRaw(writer, srs.VY); err != nil {
		return err
	}
	if err := writeG2SliceRaw(writer, srs.VZ); err != nil {
		return err
	}
	if err := writeG2SliceRaw(writer, srs.VT); err != nil {
		return err
	}
	if err := writeG2Raw(writer, srs.V); err != nil {
		return err
	}
	return writer.Flush()
}

// trustSRS skips per-point subgroup checks while loading -- safe only when
// the SRS is setup material this same process (or an equally trusted local
// pipeline) produced, never for an SRS obtained from another party. See the
// readG1SliceTrusted doc comment for why this doesn't weaken soundness in
// that case.
func readSRS(path string, trustSRS bool) (*kzh4SRS, error) {
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

	readG1s := readG1Slice
	readG2s := readG2Slice
	readG2p := readG2
	if trustSRS {
		readG1s = readG1SliceTrusted
		readG2s = readG2SliceTrusted
		readG2p = readG2Trusted
	}

	hXYZT, err := readG1s(reader)
	if err != nil {
		return nil, err
	}
	hYZT, err := readG1s(reader)
	if err != nil {
		return nil, err
	}
	hZT, err := readG1s(reader)
	if err != nil {
		return nil, err
	}
	hT, err := readG1s(reader)
	if err != nil {
		return nil, err
	}
	vX, err := readG2s(reader)
	if err != nil {
		return nil, err
	}
	vY, err := readG2s(reader)
	if err != nil {
		return nil, err
	}
	vZ, err := readG2s(reader)
	if err != nil {
		return nil, err
	}
	vT, err := readG2s(reader)
	if err != nil {
		return nil, err
	}
	v, err := readG2p(reader)
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

// readVerifierSRS loads only the SRS fields verify() actually uses: HT,
// VX, VY, VZ, V (see kzh4.go -- commit/open touch HXYZT/HYZT/HZT, verify
// never does). For any real circuit size those three skipped arrays are
// almost the entire file: HXYZT alone holds DegreeX*DegreeY*DegreeZ*DegreeT
// points, versus HT/VX/VY/VZ/V which scale with the sum of the degrees, not
// their product. Rather than decode-and-discard them, seek straight past
// their bytes -- computable exactly because the SRS uses fixed-size raw
// (uncompressed) point encoding with a 4-byte length prefix per slice, so
// each skipped section's byte length follows directly from the degree
// header read at the top of the file. This keeps verify() strict
// (subgroup-checked), matching readSRS's non-trusted path.
func readVerifierSRS(path string) (*kzh4SRS, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	header := bufio.NewReader(file)
	magic := make([]byte, len(srsMagic))
	if _, err := io.ReadFull(header, magic); err != nil {
		return nil, err
	}
	if string(magic) != srsMagic {
		return nil, fmt.Errorf("invalid SRS magic: %q", string(magic))
	}

	degreeX, err := readU64(header)
	if err != nil {
		return nil, err
	}
	degreeY, err := readU64(header)
	if err != nil {
		return nil, err
	}
	degreeZ, err := readU64(header)
	if err != nil {
		return nil, err
	}
	degreeT, err := readU64(header)
	if err != nil {
		return nil, err
	}

	const lenPrefixBytes = 4
	pointBytes := int64(bls12381.SizeOfG1AffineUncompressed)
	hxyztBytes := lenPrefixBytes + int64(degreeX)*int64(degreeY)*int64(degreeZ)*int64(degreeT)*pointBytes
	hyztBytes := lenPrefixBytes + int64(degreeY)*int64(degreeZ)*int64(degreeT)*pointBytes
	hztBytes := lenPrefixBytes + int64(degreeZ)*int64(degreeT)*pointBytes
	headerBytes := int64(len(srsMagic)) + 4*8 // magic + 4 uint64 degrees

	// Seek on the file itself (not `header`, whose bufio buffer may already
	// hold bytes past our logical read position) to an absolute offset, so
	// there's no ambiguity from buffered-but-unconsumed bytes.
	htOffset := headerBytes + hxyztBytes + hyztBytes + hztBytes
	if _, err := file.Seek(htOffset, io.SeekStart); err != nil {
		return nil, err
	}
	reader := bufio.NewReader(file)

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

// Point (de)serialization delegates to gnark-crypto's own Encoder/Decoder
// rather than hand-rolling per-point loops. This matters far more on the
// read side: decompressing a point (recovering Y from a compressed X)
// needs a field square root, so a naive serial loop over an SRS with
// millions of points dominates prove/verify wall-clock time even though
// none of that shows up in the timed section. gnark-crypto's Decoder does
// the cheap per-point byte reads serially but runs the expensive Y
// recovery through its internal worker pool (parallel.Execute), so
// swapping in the library's codec parallelizes the actual bottleneck for
// free. Compression (write side) doesn't need a square root, so it's
// cheap either way.
func writeG1(writer io.Writer, point bls12381.G1Affine) error {
	return bls12381.NewEncoder(writer).Encode(&point)
}

func readG1(reader io.Reader) (bls12381.G1Affine, error) {
	var point bls12381.G1Affine
	err := bls12381.NewDecoder(reader).Decode(&point)
	return point, err
}

func writeG2(writer io.Writer, point bls12381.G2Affine) error {
	return bls12381.NewEncoder(writer).Encode(&point)
}

func readG2(reader io.Reader) (bls12381.G2Affine, error) {
	var point bls12381.G2Affine
	err := bls12381.NewDecoder(reader).Decode(&point)
	return point, err
}

func writeG1Slice(writer io.Writer, points []bls12381.G1Affine) error {
	return bls12381.NewEncoder(writer).Encode(points)
}

func readG1Slice(reader io.Reader) ([]bls12381.G1Affine, error) {
	var points []bls12381.G1Affine
	err := bls12381.NewDecoder(reader).Decode(&points)
	return points, err
}

func writeG2Slice(writer io.Writer, points []bls12381.G2Affine) error {
	return bls12381.NewEncoder(writer).Encode(points)
}

func readG2Slice(reader io.Reader) ([]bls12381.G2Affine, error) {
	var points []bls12381.G2Affine
	err := bls12381.NewDecoder(reader).Decode(&points)
	return points, err
}

// Raw (uncompressed) variants: used only for the SRS, never for the
// artifact (opening proof), whose compressed size is a metric this
// project reports. Decode() auto-detects compressed vs. raw from each
// point's leading metadata byte, so the read side doesn't need a matching
// "raw" mode -- readG1Slice/readG2Slice/readG2 already handle both. The
// SRS is setup material, read on every prove/verify call but written once
// and never treated as "the proof", so trading disk space (~2x) to skip
// the compressed format's per-point square root entirely on every load is
// a clear win: the SRS holds millions of points, and reconstructing Y from
// a compressed X is far more expensive than just storing it, even with
// the decompression parallelized across cores.
func writeG1SliceRaw(writer io.Writer, points []bls12381.G1Affine) error {
	return bls12381.NewEncoder(writer, bls12381.RawEncoding()).Encode(points)
}

func writeG2SliceRaw(writer io.Writer, points []bls12381.G2Affine) error {
	return bls12381.NewEncoder(writer, bls12381.RawEncoding()).Encode(points)
}

func writeG2Raw(writer io.Writer, point bls12381.G2Affine) error {
	return bls12381.NewEncoder(writer, bls12381.RawEncoding()).Encode(&point)
}

// Trusted (subgroup-check-skipping) SRS readers. Subgroup checks guard
// against a point that was substituted from outside the curve's intended
// prime-order subgroup -- they say nothing about whether the SRS itself is
// a genuine, honestly-generated set of monomials (that's what a trusted
// setup ceremony is for). setup() derives every SRS point as a scalar
// multiple of a subgroup generator, so its output is in-subgroup by
// construction; a check here would only reconfirm that on every load, at
// real cost (this is most of what "SRS load" time is). Used only for
// reading the SRS (setup material produced by this same, un-adversarial
// process) -- never for the artifact/opening, which is the actual
// prover-controlled input a verifier must not trust blindly.
func readG1SliceTrusted(reader io.Reader) ([]bls12381.G1Affine, error) {
	var points []bls12381.G1Affine
	err := bls12381.NewDecoder(reader, bls12381.NoSubgroupChecks()).Decode(&points)
	return points, err
}

func readG2SliceTrusted(reader io.Reader) ([]bls12381.G2Affine, error) {
	var points []bls12381.G2Affine
	err := bls12381.NewDecoder(reader, bls12381.NoSubgroupChecks()).Decode(&points)
	return points, err
}

func readG2Trusted(reader io.Reader) (bls12381.G2Affine, error) {
	var point bls12381.G2Affine
	err := bls12381.NewDecoder(reader, bls12381.NoSubgroupChecks()).Decode(&point)
	return point, err
}
