//===- ProgramPoint.h - MLIR Program Point Class ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ProgramPoint class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_PROGRAMPOINT_H
#define MLIR_IR_PROGRAMPOINT_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir {

/// Program point represents a specific location in the execution of a program.
/// A sequence of program points can be combined into a control flow graph.
class ProgramPoint {
  public:
    /// Creates a new program point which doesn't point to anything.
    ProgramPoint() = default;

    /// Creates a new program point at the given location.
    ProgramPoint(Block *parentBlock, Block::iterator iter)
        : block(parentBlock), point(iter) {}

    /// Creates a new program point after the given operation.
    static ProgramPoint after(Operation *op) {
      return ProgramPoint(op->getBlock(), ++Block::iterator(op));
    }

    /// Creates a new program point before the given operation.
    static ProgramPoint before(Operation *op) {
      return ProgramPoint(op->getBlock(), Block::iterator(op));
    }

    /// Creates a new program point to the start of the specified block.
    static ProgramPoint start(Block *block) {
      return ProgramPoint(block, block->begin());
    }

    /// Creates a new program point to the end of the specified block.
    static ProgramPoint end(Block *block) {
      return ProgramPoint(block, block->end());
    }

    /// Reset the program point to no location.
    void clear() {
      block = nullptr;
      point = Block::iterator();
    }
   
    /// Returns true if this program point is set.
    bool isSet() const { return (block != nullptr); }

    Block *getBlock() const { return block; }
    Block::iterator getPoint() const { return point; }

  private:
    Block *block = nullptr;
    Block::iterator point;
}

} // namespace mlir

#endif // MLIR_IR_PROGRAMPOINT_H
