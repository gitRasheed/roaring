// +build arm64,!gccgo,!appengine

#include "textflag.h"

// NEON-vectorized main loop for intersection2by2Cardinality.
// All-pairs 8x8 block compare: the 8 lanes of set1's block are matched
// against all 8 rotations of set2's block (NEON has no cmpestrm), giving a
// per-lane match mask; subtracting the mask (0 or -1 lanes) from a 16-bit
// accumulator counts matches. Blocks advance by the smaller maximum, both
// on ties. A two-way range gate with predicted branches fast-forwards runs
// of non-overlapping blocks on scalar boundary loads alone, without
// touching the vector pipe. Lane counts cannot overflow: at most 8192
// blocks of ones per input. Processes full blocks; the Go wrapper drains
// the tails.
//
// Register use:
//   R0/R1 cursors into set1/set2, R3/R4 their full-block end addresses,
//   R8/R9 set1 block's first/last value, R10/R11 same for set2.
//   V0/V1 current set1/set2 blocks, V2 the match accumulator, V3-V7 temps.

// Match mask of V0's lanes against all rotations of V1, result in V3.
#define MATCH8 \
	VCMEQ V1.H8, V0.H8, V3.H8          \
	VEXT  $2, V1.B16, V1.B16, V4.B16   \
	VCMEQ V4.H8, V0.H8, V4.H8          \
	VEXT  $4, V1.B16, V1.B16, V5.B16   \
	VCMEQ V5.H8, V0.H8, V5.H8          \
	VEXT  $6, V1.B16, V1.B16, V6.B16   \
	VCMEQ V6.H8, V0.H8, V6.H8          \
	VORR  V4.B16, V3.B16, V3.B16       \
	VORR  V6.B16, V5.B16, V5.B16       \
	VEXT  $8, V1.B16, V1.B16, V4.B16   \
	VCMEQ V4.H8, V0.H8, V4.H8          \
	VEXT  $10, V1.B16, V1.B16, V6.B16  \
	VCMEQ V6.H8, V0.H8, V6.H8          \
	VORR  V6.B16, V4.B16, V4.B16       \
	VEXT  $12, V1.B16, V1.B16, V6.B16  \
	VCMEQ V6.H8, V0.H8, V6.H8          \
	VEXT  $14, V1.B16, V1.B16, V7.B16  \
	VCMEQ V7.H8, V0.H8, V7.H8          \
	VORR  V7.B16, V6.B16, V6.B16       \
	VORR  V5.B16, V3.B16, V3.B16       \
	VORR  V6.B16, V4.B16, V4.B16       \
	VORR  V4.B16, V3.B16, V3.B16

// func intersectCardKernelNEON(set1, set2 []uint16) (card, pos1, pos2 int)
TEXT ·intersectCardKernelNEON(SB), NOSPLIT, $0-72
	MOVD set1_base+0(FP), R0
	MOVD set2_base+24(FP), R1
	MOVD set1_len+8(FP), R3
	MOVD set2_len+32(FP), R4

	// end addresses over whole blocks: base + (len &^ 7) * 2
	AND $~7, R3, R3
	AND $~7, R4, R4
	ADD R3<<1, R0, R3
	ADD R4<<1, R1, R4

	VEOR  V2.B16, V2.B16, V2.B16
	VLD1  (R0), [V0.H8]
	VLD1  (R1), [V1.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	MOVHU (R1), R10
	MOVHU 14(R1), R11

loop:
	// Range gate, strict <: a block ending on b's first value may match it.
	CMP  R10, R9
	BLO  ffA
	CMP  R8, R11
	BLO  ffB

	MATCH8
	VSUB V3.H8, V2.H8, V2.H8

	// Advance the block with the smaller maximum; both when equal. If set1
	// runs out first here, pos2 still names the consumed set2 block; the
	// tail rescan is harmless since set1's tail exceeds that block's max.
	CMP R11, R9
	BHI advB
	BLO advA
	ADD $16, R0
	CMP R3, R0
	BHS done
	ADD $16, R1
	CMP R4, R1
	BHS done
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	MOVHU 14(R1), R11
	B loop

advA:
	ADD $16, R0
	CMP R3, R0
	BHS done
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	B loop

advB:
	ADD $16, R1
	CMP R4, R1
	BHS done
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	MOVHU 14(R1), R11
	B loop

// Fast-forward set1 past blocks wholly below set2's block: scalar boundary
// loads only, one predicted branch per skipped block.
ffA:
	ADD $16, R0
	CMP R3, R0
	BHS done
	MOVHU 14(R0), R9
	CMP   R10, R9
	BLO   ffA
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	B loop

ffB:
	ADD $16, R1
	CMP R4, R1
	BHS done
	MOVHU 14(R1), R11
	CMP   R8, R11
	BLO   ffB
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	B loop

done:
	VUADDLV V2.H8, V2
	VMOV    V2.D[0], R5
	MOVD    R5, card+48(FP)

	MOVD set1_base+0(FP), R8
	SUB  R8, R0, R0
	LSR  $1, R0, R0
	MOVD R0, pos1+56(FP)
	MOVD set2_base+24(FP), R8
	SUB  R8, R1, R1
	LSR  $1, R1, R1
	MOVD R1, pos2+64(FP)
	RET
