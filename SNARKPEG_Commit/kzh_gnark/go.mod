module zkcnn_kzh_gnark

go 1.24.1

require (
	github.com/bits-and-blooms/bitset v1.20.0 // indirect
	github.com/consensys/gnark-crypto v0.19.1
	golang.org/x/sys v0.30.0 // indirect
)

replace github.com/bits-and-blooms/bitset => ../third_party_go/bitset

replace golang.org/x/sys => ../third_party_go/xsys
