package main

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
)

func main() {
	if err := run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) < 2 {
		printUsage(args)
		return errors.New("missing command")
	}

	switch args[1] {
	case "setup":
		return commandSetup(args[2:])
	case "prove":
		return commandProve(args[2:])
	case "verify":
		return commandVerify(args[2:])
	default:
		printUsage(args)
		return fmt.Errorf("unknown command: %s", args[1])
	}
}

func commandSetup(args []string) error {
	parsed, err := parseArgs(args)
	if err != nil {
		return err
	}
	numVarsStr, err := requiredArg(parsed, "--num-vars")
	if err != nil {
		return err
	}
	numVars, err := strconv.Atoi(numVarsStr)
	if err != nil {
		return fmt.Errorf("invalid --num-vars: %w", err)
	}
	srsPath, err := requiredArg(parsed, "--srs")
	if err != nil {
		return err
	}

	srs, err := setup(numVars)
	if err != nil {
		return err
	}
	return writeSRS(srsPath, srs)
}

func commandProve(args []string) error {
	parsed, err := parseArgs(args)
	if err != nil {
		return err
	}
	srsPath, err := requiredArg(parsed, "--srs")
	if err != nil {
		return err
	}
	polyPath, err := requiredArg(parsed, "--poly")
	if err != nil {
		return err
	}
	pointPath, err := requiredArg(parsed, "--point")
	if err != nil {
		return err
	}
	artifactPath, err := requiredArg(parsed, "--artifact")
	if err != nil {
		return err
	}
	metricsPath, err := requiredArg(parsed, "--metrics")
	if err != nil {
		return err
	}

	loadStart := time.Now()
	// Trusted: this SRS is setup material produced by this same pipeline
	// (see readSRS's doc comment), not an input from another party.
	srs, err := readSRS(srsPath, true)
	if err != nil {
		return err
	}
	srsLoadElapsed := time.Since(loadStart).Seconds()
	polyValues, err := readScalarVectorCxx(polyPath)
	if err != nil {
		return err
	}
	point, err := readScalarVectorCxx(pointPath)
	if err != nil {
		return err
	}
	reverseScalars(point)
	poly := newMultilinearPolynomial(polyValues)
	inputLoadElapsed := time.Since(loadStart).Seconds() - srsLoadElapsed

	start := time.Now()
	commitment, aux, err := commit(srs, poly)
	if err != nil {
		return err
	}
	opening, err := open(srs, point, aux, poly)
	if err != nil {
		return err
	}
	elapsed := time.Since(start).Seconds()

	if err := writeArtifact(artifactPath, commitment, opening); err != nil {
		return err
	}
	commitmentSize, openingSize := proofSizeBytes(commitment, opening)
	return writeMetrics(metricsPath, map[string]string{
		"prove_time_sec":        formatFloat(elapsed),
		"srs_load_time_sec":     formatFloat(srsLoadElapsed),
		"input_load_time_sec":   formatFloat(inputLoadElapsed),
		"proof_size_bytes":      strconv.FormatUint(commitmentSize+openingSize, 10),
		"commitment_size_bytes": strconv.FormatUint(commitmentSize, 10),
		"opening_size_bytes":    strconv.FormatUint(openingSize, 10),
	})
}

func commandVerify(args []string) (retErr error) {
	parsed, err := parseArgs(args)
	if err != nil {
		return err
	}
	srsPath, err := requiredArg(parsed, "--srs")
	if err != nil {
		return err
	}
	pointPath, err := requiredArg(parsed, "--point")
	if err != nil {
		return err
	}
	valuePath, err := requiredArg(parsed, "--value")
	if err != nil {
		return err
	}
	artifactPath, err := requiredArg(parsed, "--artifact")
	if err != nil {
		return err
	}
	metricsPath, err := requiredArg(parsed, "--metrics")
	if err != nil {
		return err
	}

	loadStart := time.Now()
	// verify() only ever reads HT/VX/VY/VZ/V (see kzh4.go), never
	// HXYZT/HYZT/HZT -- readVerifierSRS seeks past those instead of
	// decoding them, and keeps subgroup checks on for the fields it does
	// read (unlike prove, a verifier is the party meant to distrust its
	// inputs, and may in the future be pointed at an SRS this process
	// didn't generate itself).
	srs, err := readVerifierSRS(srsPath)
	if err != nil {
		return err
	}
	srsLoadElapsed := time.Since(loadStart).Seconds()
	point, err := readScalarVectorCxx(pointPath)
	if err != nil {
		return err
	}
	reverseScalars(point)
	value, err := readScalarValueCxx(valuePath)
	if err != nil {
		return err
	}
	commitment, opening, err := readArtifact(artifactPath)
	if err != nil {
		return err
	}
	inputLoadElapsed := time.Since(loadStart).Seconds() - srsLoadElapsed

	start := time.Now()
	defer func() {
		if r := recover(); r != nil {
			retErr = fmt.Errorf("panic during KZH4 verify: %v", r)
		}
		if metricsErr := writeMetrics(metricsPath, map[string]string{
			"verify_time_sec":     formatFloat(time.Since(start).Seconds()),
			"srs_load_time_sec":   formatFloat(srsLoadElapsed),
			"input_load_time_sec": formatFloat(inputLoadElapsed),
		}); metricsErr != nil && retErr == nil {
			retErr = metricsErr
		}
	}()

	if err := verify(srs, point, &value, commitment, opening); err != nil {
		return err
	}
	return nil
}

func parseArgs(args []string) (map[string]string, error) {
	out := make(map[string]string)
	for i := 0; i < len(args); i += 2 {
		if i+1 >= len(args) {
			return nil, fmt.Errorf("missing value for %s", args[i])
		}
		if len(args[i]) < 3 || args[i][:2] != "--" {
			return nil, fmt.Errorf("unexpected argument: %s", args[i])
		}
		out[args[i]] = args[i+1]
	}
	return out, nil
}

func requiredArg(args map[string]string, key string) (string, error) {
	value, ok := args[key]
	if !ok {
		return "", fmt.Errorf("missing %s", key)
	}
	return value, nil
}

func reverseScalars(values []fr.Element) {
	for i, j := 0, len(values)-1; i < j; i, j = i+1, j-1 {
		values[i], values[j] = values[j], values[i]
	}
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func printUsage(args []string) {
	program := "zkcnn_kzh_cli"
	if len(args) > 0 && args[0] != "" {
		program = args[0]
	}
	fmt.Fprintf(os.Stderr, "Usage:\n  %s setup --num-vars <n> --srs <path>\n  %s prove --srs <path> --poly <path> --point <path> --artifact <path> --metrics <path>\n  %s verify --srs <path> --point <path> --value <path> --artifact <path> --metrics <path>\n", program, program, program)
}
