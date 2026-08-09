//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

//go:noescape
func intersectCardKernelNEON(set1, set2 []uint16) (card, pos1, pos2 int)

//go:noescape
func intersectKernelNEON(set1, set2, buffer []uint16, shuf *byte, spill *[8]uint16) (outLen, pos1, pos2, spilled int)

// Below this the kernel setup usually loses to the scalar path.
// Duplicate values get unspecified results; stores stay within capacity.
const (
	neonIntersectCardThreshold = 16
	neonIntersectThreshold     = 16
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
	// andArray passes len 0; COW cloning keeps iandArray's spare cap private.
	buffer = buffer[:cap(buffer)]
	var spill [8]uint16
	outLen, pos1, pos2, spilled := intersectKernelNEON(set1, set2, buffer, &uniqshuf[0], &spill)
	if spilled != 0 {
		// set1 may alias buffer: drain from this copy, never reread.
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
	// The tails were never compared; sorted sets keep the completion exact.
	return card + localintersect2by2Cardinality(set1[pos1:], set2[pos2:])
}
