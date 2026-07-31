//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

//go:noescape
func union2by2scalar(set1 []uint16, set2 []uint16, buffer []uint16) (size int)

//go:noescape
func unionKernelNEON(set1, set2, buffer []uint16, shuf *byte, leftover *[16]uint16) (outLen, pos1, pos2, leftoverLen int)

// uniqshuf[m] is the 16-byte TBL index vector compacting the lanes NOT set in
// mask m to the front (0xFF elsewhere; TBL yields 0 for out-of-range).
// Identical layout to CRoaring's table (generated there by simdunion.py).
var uniqshuf [256 * 16]byte

func init() {
	for m := 0; m < 256; m++ {
		pos := 0
		for lane := 0; lane < 8; lane++ {
			if m&(1<<lane) == 0 {
				uniqshuf[m*16+pos*2] = byte(2 * lane)
				uniqshuf[m*16+pos*2+1] = byte(2*lane + 1)
				pos++
			}
		}
		for ; pos < 8; pos++ {
			uniqshuf[m*16+pos*2] = 0xFF
			uniqshuf[m*16+pos*2+1] = 0xFF
		}
	}
}

// Dispatch to the vector kernel only when both inputs reach this size:
// measured on Graviton2 (Neoverse-N1) and Graviton4 (Neoverse-V2) with
// varied-data benchmarks, 1024 is the first size where the vector path wins
// decisively (2.3-3.5x on random shapes) while smaller inputs favor the
// scalar loop on wide cores.
const neonUnionThreshold = 1024

func union2by2(set1 []uint16, set2 []uint16, buffer []uint16) int {
	if len(set1) < neonUnionThreshold || len(set2) < neonUnionThreshold {
		return union2by2scalar(set1, set2, buffer)
	}
	// Callers (e.g. lazyorArray) may pass a zero-length buffer with spare
	// capacity: the historical asm contract only uses the base pointer and
	// the caller reslices to the returned size afterwards.
	buffer = buffer[:cap(buffer)]
	var leftover [16]uint16
	outLen, pos1, pos2, ll := unionKernelNEON(set1, set2, buffer, &uniqshuf[0], &leftover)
	// The carry leftovers and the exhausted input's tail are two sorted runs;
	// merge them, then merge that with the other input's remainder.
	var tmp [16]uint16
	if pos1 == len(set1)/8 {
		m := scalarMergeUnion(leftover[:ll], set1[8*pos1:], tmp[:])
		outLen += mergeUnionLookahead(tmp[:m], set2[8*pos2:], buffer[outLen:])
	} else {
		m := scalarMergeUnion(leftover[:ll], set2[8*pos2:], tmp[:])
		outLen += mergeUnionLookahead(tmp[:m], set1[8*pos1:], buffer[outLen:])
	}
	return outLen
}

// mergeUnionLookahead is scalarMergeUnion with b read in 8-element chunks
// ahead of the writes. iorArray passes a buffer whose backing array holds
// set1 above the write region; tail writes can lead the b reads by up to 7
// elements there, so reading 8 ahead keeps every element read before it can
// be overwritten.
func mergeUnionLookahead(a, b, out []uint16) int {
	i, k := 0, 0
	for j := 0; j < len(b); {
		var w [8]uint16
		c := copy(w[:], b[j:])
		j += c
		for x := 0; x < c; x++ {
			for i < len(a) && a[i] < w[x] {
				out[k] = a[i]
				i++
				k++
			}
			if i < len(a) && a[i] == w[x] {
				i++
			}
			out[k] = w[x]
			k++
		}
	}
	for ; i < len(a); i++ {
		out[k] = a[i]
		k++
	}
	return k
}

// scalarMergeUnion merges two sorted duplicate-free slices into out,
// dropping cross-slice duplicates, and returns the number written.
func scalarMergeUnion(a, b, out []uint16) int {
	i, j, k := 0, 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			out[k] = a[i]
			i++
		case b[j] < a[i]:
			out[k] = b[j]
			j++
		default:
			out[k] = a[i]
			i++
			j++
		}
		k++
	}
	k += copy(out[k:], a[i:])
	k += copy(out[k:], b[j:])
	return k
}
