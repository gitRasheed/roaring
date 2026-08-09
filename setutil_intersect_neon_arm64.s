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

	// If set1 ends on a tie, pos2 retains the consumed set2 block; sorted
	// tails make rescanning it safe.
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

// func intersectKernelNEON(set1, set2, buffer []uint16, shuf *byte, spill *[8]uint16) (outLen, pos1, pos2, spilled int)
//
// Materializing variant. Matched set1 lanes are compacted with a
// mask-indexed TBL and stored. Two store safety conditions (both must
// hold for a 16-byte store, else an exact 4/2/1 prefix store runs):
//   out+8 <= cap: andArray may allocate exactly min(len1,len2);
//   out <= pos1: iandArray aliases buffer with unread set1 data.
// On exit, if set1's current block was loaded but not fully drained (the
// loop stopped on set2's side), it is spilled for the Go wrapper: its
// memory may already hold compacted output in the aliased case.
//
// Register use as the cardinality kernel (V2 unused), plus: R2 out cursor,
// R5 out base, R6 shuf table, R7 out capacity end address, R19 set1 base,
// R12 = 0x0101010101010101, R13 = 0x0102040810204080, R14-R17 mask/count
// extraction and store scratch.
TEXT ·intersectKernelNEON(SB), NOSPLIT, $0-120
	MOVD set1_base+0(FP), R0
	MOVD set2_base+24(FP), R1
	MOVD buffer_base+48(FP), R2
	MOVD set1_len+8(FP), R3
	MOVD set2_len+32(FP), R4
	MOVD buffer_cap+64(FP), R7
	MOVD shuf+72(FP), R6

	MOVD R0, R19
	MOVD R2, R5
	ADD  R7<<1, R2, R7

	AND $~7, R3, R3
	AND $~7, R4, R4
	ADD R3<<1, R0, R3
	ADD R4<<1, R1, R4

	MOVD $0x0101010101010101, R12
	MOVD $0x0102040810204080, R13

	// uniqshuf keeps lanes clear in its index; indexing by 255-mask from
	// the last entry keeps exactly the matched lanes.
	ADD $4080, R6, R6

	VLD1  (R0), [V0.H8]
	VLD1  (R1), [V1.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	MOVHU (R1), R10
	MOVHU 14(R1), R11

mloop:
	CMP  R10, R9
	BLO  mffA
	CMP  R8, R11
	BLO  mffB

	MATCH8

	VUZP1 V3.B16, V3.B16, V4.B16
	VMOV  V4.D[0], R14
	AND   R12, R14, R14
	MUL   R13, R14, R15
	LSR   $56, R15, R15      // mask
	MUL   R12, R14, R14
	LSR   $56, R14, R14      // count
	CBZ   R15, madv

	SUB  R15<<4, R6, R16
	VLD1 (R16), [V4.B16]
	VTBL V4.B16, [V0.B16], V4.B16

	// A full store must fit the capacity and not overtake unread set1 data.
	ADD  $16, R2, R16
	CMP  R7, R16
	BHI  mprefix
	SUB  R5, R2, R16         // out bytes
	SUB  R19, R0, R17        // pos1 bytes
	CMP  R17, R16
	BHI  mprefix
	VST1 [V4.B16], (R2)
	ADD  R14<<1, R2, R2
	B    madv

mprefix:
	// Strictly increasing inputs cannot reach here with count 8 (which the
	// 4/2/1 groups store as nothing); duplicate-bearing inputs can, so
	// clamp the count to the remaining capacity.
	SUB  R2, R7, R16
	LSR  $1, R16, R16
	CMP  R16, R14
	CSEL LS, R14, R16, R14
	VMOV V4.D[0], R16
	VMOV V4.D[1], R17
	TBZ  $2, R14, mpref2
	MOVD R16, (R2)
	ADD  $8, R2
	MOVD R17, R16
mpref2:
	TBZ  $1, R14, mpref1
	MOVW R16, (R2)
	ADD  $4, R2
	LSR  $32, R16, R16
mpref1:
	TBZ  $0, R14, madv
	MOVH R16, (R2)
	ADD  $2, R2

madv:
	CMP R11, R9
	BHI madvB
	BLO madvA
	// Advance set2 before either exit so pos2 cannot reference memory the
	// store path rewrote (self-intersection aliases all three slices).
	ADD $16, R0
	ADD $16, R1
	CMP R3, R0
	BHS mdoneNoSpill
	CMP R4, R1
	// The next set1 block was never loaded, so no spill is needed.
	BHS mdoneNoSpill
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	MOVHU 14(R1), R11
	B mloop

madvA:
	ADD $16, R0
	CMP R3, R0
	BHS mdoneNoSpill
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	MOVHU 14(R0), R9
	B mloop

madvB:
	ADD $16, R1
	CMP R4, R1
	BHS mdoneSpill
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	MOVHU 14(R1), R11
	B mloop

mffA:
	ADD $16, R0
	CMP R3, R0
	BHS mdoneNoSpill
	MOVHU 14(R0), R9
	CMP   R10, R9
	BLO   mffA
	VLD1  (R0), [V0.H8]
	MOVHU (R0), R8
	B mloop

mffB:
	ADD $16, R1
	CMP R4, R1
	BHS mdoneSpill
	MOVHU 14(R1), R11
	CMP   R8, R11
	BLO   mffB
	VLD1  (R1), [V1.H8]
	MOVHU (R1), R10
	B mloop

mdoneSpill:
	// Hand the register copy to the wrapper; the block's memory may
	// already hold compacted output.
	MOVD spill+80(FP), R16
	VST1 [V0.H8], (R16)
	MOVD $1, R16
	MOVD R16, spilled+112(FP)
	ADD  $16, R0
	B    mout

mdoneNoSpill:
	MOVD ZR, spilled+112(FP)

mout:
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
