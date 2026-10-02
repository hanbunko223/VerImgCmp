import CircuitCorrectness.ByteCertificates.Base
namespace CircuitCorrectness.ConcreteBytes
def expectedPixelChunk (c : Nat) : Array Nat :=
  ((List.range 128).map (fun i => valueWire (128*c+i))).toArray

end CircuitCorrectness.ConcreteBytes
