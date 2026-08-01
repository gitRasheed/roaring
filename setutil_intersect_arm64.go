//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

//go:noescape
func intersectCardKernelNEON(set1, set2 []uint16) (card, pos1, pos2 int)

// Below this size the scalar path wins (BenchmarkIntersectCard, Graviton 4).
const neonIntersectCardThreshold = 64

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
