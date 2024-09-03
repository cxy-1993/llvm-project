// RUN: mlir-opt %s --lower-affine 
#map = affine_map<(d0) -> (d0)>
func.func @matmul(%A: memref<4096xf32, 101>, %B: memref<4096xf32, 101>, %C: memref<4096xf32, 101>) -> memref<4096xf32, 101> {
  affine.for %i = 0 to 4096 step 128 {
    affine.for %ii = #map(%i) to #map(%i) {
      %5 = memref.load %A[%ii] : memref<4096xf32, 101>
      %6 = memref.load %B[%ii] : memref<4096xf32, 101>
      %7 = memref.load %C[%ii] : memref<4096xf32, 101>
      %8 = arith.mulf %5, %6 : f32 
      %9 = arith.addf %7, %8 : f32 
      memref.store %9, %C[%ii] : memref<4096xf32, 101>
    }   
  }
  return %C : memref<4096xf32, 101>
}
