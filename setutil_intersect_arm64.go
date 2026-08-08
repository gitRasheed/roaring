//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

//go:noescape
func intersectCardKernelNEON(set1, set2 []uint16) (card, pos1, pos2 int)

//go:noescape
func intersectKernelNEON(set1, set2, buffer []uint16, shuf *byte, spill *[8]uint16) (outLen, pos1, pos2, spilled int)

// The kernel compacts the lanes SET in its match mask, the complement of
// the union kernel's dedup masks, so it reuses uniqshuf via an inverted
// index (uniqshuf[m^0xFF] keeps exactly the lanes set in m).

// The kernels usually win from n=8..16; 32 keeps margin on unmeasured cores.
// Inputs are sorted sets. Arrays with duplicate values (Validate accepts
// adjacent equals) get unspecified results, but stores stay within the
// buffer's capacity.
const (
	neonIntersectCardThreshold = 32
	neonIntersectThreshold     = 32
)

func intersection2by2(
	set1 []uint16,
	set2 []uint16,
	buffer []uint16,
) int {
	if len(set1)*64 < len(set2) {
		return onesidedgallopingintersect2by2(set1, set2, buffer)
	} else if len(set2)*64 < len(set1) {
		return onesidedgallopingintersect2by2(set2, set1, buffer)
	}
	if len(set1) < neonIntersectThreshold || len(set2) < neonIntersectThreshold {
		return localintersect2by2(set1, set2, buffer)
	}
	if set1[len(set1)-1] < set2[0] || set2[len(set2)-1] < set1[0] {
		return 0
	}
	// Callers may pass a zero-length buffer with capacity (andArray). The
	// spare capacity is never shared memory: copy-on-write containers are
	// cloned before iand.
	buffer = buffer[:cap(buffer)]
	var spill [8]uint16
	outLen, pos1, pos2, spilled := intersectKernelNEON(set1, set2, buffer, &uniqshuf[0], &spill)
	if spilled != 0 {
		// The kernel handed back set1's retained block: in the aliased
		// iandArray case its memory may already hold compacted output,
		// so it must be drained from this copy, never reread.
		i := 0
		for i < 8 && pos2 < len(set2) {
			switch {
			case spill[i] < set2[pos2]:
				i++
			case set2[pos2] < spill[i]:
				pos2++
			default:
				buffer[outLen] = spill[i]
				outLen++
				i++
				pos2++
			}
		}
	}
	return outLen + localintersect2by2(set1[pos1:], set2[pos2:], buffer[outLen:])
}

// intersects2by2 (the boolean variant) stays scalar: its early exit on the
// first match beats fixed-work vector blocks on typical inputs.

// intersection2by2Cardinality computes the cardinality of the intersection
func intersection2by2Cardinality(
	set1 []uint16,
	set2 []uint16,
) int {
	if len(set1)*64 < len(set2) {
		return onesidedgallopingintersect2by2Cardinality(set1, set2)
	} else if len(set2)*64 < len(set1) {
		return onesidedgallopingintersect2by2Cardinality(set2, set1)
	}
	if len(set1) < neonIntersectCardThreshold || len(set2) < neonIntersectCardThreshold {
		return localintersect2by2Cardinality(set1, set2)
	}
	if set1[len(set1)-1] < set2[0] || set2[len(set2)-1] < set1[0] {
		return 0
	}
	card, pos1, pos2 := intersectCardKernelNEON(set1, set2)
	// The kernel stops when either side has no full block left; the tail
	// pairs were never compared, so a scalar pass over them completes the
	// count without double-counting (each value occurs once per input).
	return card + localintersect2by2Cardinality(set1[pos1:], set2[pos2:])
}
