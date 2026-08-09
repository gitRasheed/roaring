// +build arm64,!gccgo,!appengine

#include "textflag.h"

// NEON-vectorized main loop for difference (andnot).
// All-pairs 8x8 block compare as in the intersection kernel, with DEFERRED
// EMISSION: a lane of set1's block may match any later set2 block while the
// block is retained, so per-block match masks are OR-accumulated in V2 and
// the unmatched lanes are compacted and stored only when the block retires
// (set1 advance, tie, fast-forward, or exit). Output is a subset of the
// consumed set1 prefix, so out <= pos1 in elements at every retirement and
// the full 16-byte store always fits: the buffer holds len(set1) elements
// and a loaded block means pos1+8 <= len(set1). This holds for any input
// the callers can pass, so there is no prefix store and no clamp.
// Blocks advance by the smaller maximum, both on ties. The two-way range
// gate fast-forwards non-overlapping runs; skipped set1 blocks are stored
// whole (nothing left in set2 can match them), loaded before stored so an
// aliased buffer never overtakes the read.
//
// Register use:
//   R0/R1 cursors into set1/set2, R3/R4 their full-block end addresses,
//   R8/R9 set1 block's first/last value, R10/R11 same for set2. R2 out
//   cursor, R5 out base, R6 shuf table, R19 set1 base,
//   R12 = 0x0101010101010101, R13 = 0x0102040810204080, R14-R17 scratch.
//   V0/V1 current set1/set2 blocks, V2 accumulated match mask, V3-V7 temps.

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

// Retire V0: compact the lanes clear in V2's accumulated mask (uniqshuf
// keeps lanes clear in its index, andnot's exact polarity), store, advance
// out by the survivor count, reset V2. A full-mask block stores 16 bytes of
// table zeros and advances out by nothing; the bytes land below pos1 in
// already-consumed territory.
#define EMITBLOCK \
	VUZP1 V2.B16, V2.B16, V4.B16       \
	VMOV  V4.D[0], R14                 \
	AND   R12, R14, R14                \
	MUL   R12, R14, R16                \
	LSR   $56, R16, R16                \
	MUL   R13, R14, R15                \
	LSR   $56, R15, R15                \
	ADD   R15<<4, R6, R17              \
	VLD1  (R17), [V4.B16]              \
	VTBL  V4.B16, [V0.B16], V4.B16     \
	VST1  [V4.B16], (R2)               \
	ADD   $16, R2                      \
	SUB   R16<<1, R2, R2               \
	VEOR  V2.B16, V2.B16, V2.B16

// func andnotKernelNEON(set1, set2, buffer []uint16, shuf *byte, spill *[8]uint16) (outLen, pos1, pos2, spilled, accMask int)
TEXT ·andnotKernelNEON(SB), NOSPLIT, $0-128
	MOVD set1_base+0(FP), R0
	MOVD set2_base+24(FP), R1
	MOVD buffer_base+48(FP), R2
	MOVD set1_len+8(FP), R3
	MOVD set2_len+32(FP), R4
	MOVD shuf+72(FP), R6

	MOVD R0, R19
	MOVD R2, R5

	AND $~7, R3, R3
	AND $~7, R4, R4
	ADD R3<<1, R0, R3
	ADD R4<<1, R1, R4

	MOVD $0x0101010101010101, R12
	MOVD $0x0102040810204080, R13

	VEOR  V2.B16, V2.B16, V2.B16
	VLD1  (R0), [V0.H8]
	VLD1  (R1), [V1.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	MOVHU (R1), R10
	MOVHU 14(R1), R11

nloop:
	// Range gate, strict <: a block ending on the other's first value may match.
	CMP  R10, R9
	BLO  nffA
	CMP  R8, R11
	BLO  nffB

	MATCH8
	VORR V3.B16, V2.B16, V2.B16

	// A tie exit may retain a consumed set2 block; rescanning it is safe.
	CMP R11, R9
	BHI nadvB
	BLO nadvA
	EMITBLOCK
	// Advance set2 first so pos2 never points at rewritten memory.
	ADD $16, R0
	ADD $16, R1
	CMP R3, R0
	BHS ndoneNoSpill
	CMP R4, R1
	// The next set1 block was never loaded, so no spill is needed.
	BHS ndoneNoSpill
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	MOVHU 14(R1), R11
	B nloop

nadvA:
	EMITBLOCK
	ADD $16, R0
	CMP R3, R0
	BHS ndoneNoSpill
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	B nloop

nadvB:
	ADD $16, R1
	CMP R4, R1
	BHS ndoneSpill
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	MOVHU 14(R1), R11
	B nloop

nffA:
	// Nothing left in set2 can match this or any wholly-lower set1 block:
	// retire the current block, then copy skipped blocks through whole.
	EMITBLOCK
nffAskip:
	ADD $16, R0
	CMP R3, R0
	BHS ndoneNoSpill
	MOVHU 14(R0), R9
	CMP   R10, R9
	BHS   nffAload
	VLD1  (R0), [V0.H8]
	VST1  [V0.H8], (R2)
	ADD   $16, R2
	B     nffAskip

nffAload:
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	B nloop

nffB:
	ADD $16, R1
	CMP R4, R1
	BHS ndoneSpill
	MOVHU 14(R1), R11
	CMP   R8, R11
	BLO   nffB
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	B nloop

ndoneSpill:
	// Hand back the register copy and its accumulated mask; the block's
	// memory may hold output.
	MOVD  spill+80(FP), R16
	VST1  [V0.H8], (R16)
	VUZP1 V2.B16, V2.B16, V4.B16
	VMOV  V4.D[0], R14
	AND   R12, R14, R14
	MUL   R13, R14, R15
	LSR   $56, R15, R15
	MOVD  R15, accMask+120(FP)
	MOVD  $1, R16
	MOVD  R16, spilled+112(FP)
	ADD   $16, R0
	B     nout

ndoneNoSpill:
	MOVD ZR, spilled+112(FP)
	MOVD ZR, accMask+120(FP)

nout:
	SUB  R5, R2, R2
	LSR  $1, R2, R2
	MOVD R2, outLen+88(FP)
	SUB  R19, R0, R0
	LSR  $1, R0, R0
	MOVD R0, pos1+96(FP)
	MOVD set2_base+24(FP), R8
	SUB  R8, R1, R1
	LSR  $1, R1, R1
	MOVD R1, pos2+104(FP)
	RET
