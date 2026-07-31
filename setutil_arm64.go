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
// Initialized via a variable initializer, not init(): package-level variable
// initializers in other files run before init() functions, and one reaching
// union2by2 would silently read a zero table.
var uniqshuf = buildUniqshuf()

func buildUniqshuf() (t [256 * 16]byte) {
	for m := 0; m < 256; m++ {
		pos := 0
		for lane := 0; lane < 8; lane++ {
			if m&(1<<lane) == 0 {
				t[m*16+pos*2] = byte(2 * lane)
				t[m*16+pos*2+1] = byte(2*lane + 1)
				pos++
			}
		}
		for ; pos < 8; pos++ {
			t[m*16+pos*2] = 0xFF
			t[m*16+pos*2+1] = 0xFF
		}
	}
	return t
}

// Dispatch to the vector kernel only when both inputs reach this size.
// Measured (varied-data, forced-kernel sweeps on Graviton 2 and 4 after the
// bulk-copy tail fix): Neoverse-N1 has no crossover in range — the kernel
// wins from n=128 (1.7-2.5x by 256-384) — while the wider Neoverse-V2
// crosses over near 384 (>=1.02x there, 3-4x by 2048). 384 captures the
// N1 wins at a worst case of about -11% on density-mismatched inputs on
// V2. High-overlap inputs stay ~0.56x at every size on V2; no size gate
// can detect that shape. Measured crossovers are upper bounds: fixed
// per-variant data still lets the scalar path's predictor memorize.
const neonUnionThreshold = 384

func union2by2(set1 []uint16, set2 []uint16, buffer []uint16) int {
	if len(set1) < neonUnionThreshold || len(set2) < neonUnionThreshold {
		return union2by2scalar(set1, set2, buffer)
	}
	return unionNEON(set1, set2, buffer)
}

// unionNEON is the vector kernel plus its scalar tails; both inputs must
// have at least 8 elements. Split from union2by2 so tests exercise the
// exact shipped tail logic regardless of the dispatch threshold.
func unionNEON(set1 []uint16, set2 []uint16, buffer []uint16) int {
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
		// a (the small carry/tail merge) usually drains within a few
		// elements; bulk-copy the rest of b instead of walking it.
		// copy is memmove and the write cursor never passes the read
		// frontier in the aliased iorArray geometry, so this is safe.
		if i == len(a) {
			return k + copy(out[k:], b[j:])
		}
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
