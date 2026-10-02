import CircuitCorrectness.DctProgramCertificates.Base
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000
namespace CircuitCorrectness.DctProgramCertificates.Tests
open ConcreteBytes.Codes

/-- A changed coefficient code is rejected against the actual first exported DCT row. -/
example : (row ExportedData.chunk540.2).1 ≠
    { firstRow 0 0 0 with
      a := (4,matrixCode 0 0+1) :: (firstRow 0 0 0).a.tail } := by decide

/-- The constant-one input wire is part of the checked row correspondence. -/
example : (row ExportedData.chunk540.2).1 ≠
    { firstRow 0 0 0 with b := [(97633,0)] } := by decide

/-- The actual allocated first-stage destination cannot be shifted. -/
example : (row ExportedData.chunk540.2).1 ≠
    { firstRow 0 0 0 with a := (firstRow 0 0 0).a.map fun (i,c) =>
      (if i = 69124 then 69125 else i,c) } := by decide

/-- Retaining the same accumulator index but swapping the first RGB coefficient
fails the row certificate; canonical coefficient ordering is checked. -/
example : (row ExportedData.chunk580.2).1 ≠ hornerAt 0 (0,0,1) := by decide

/-- A shifted accumulator destination also fails the concrete Horner certificate. -/
example : (row ExportedData.chunk580.2).1 ≠
    { hornerAt 0 (0,0,0) with c := (hornerAt 0 (0,0,0)).c.map fun (i,c) =>
      (if i = 74244 then 74245 else i,c) } := by decide

end CircuitCorrectness.DctProgramCertificates.Tests
