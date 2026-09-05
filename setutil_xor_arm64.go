//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

//go:noescape
func xorKernelNEON(set1, set2, buffer []uint16, shuf *byte) (outLen, pos1, pos2 int)

// The kernel needs 16 unread elements on both sides and a buffer that does not
// alias either input, with cap(buffer) >= len(set1)+len(set2).
// Below 32 its setup usually loses to the scalar path.
// Duplicate values get unspecified results; stores stay within capacity.
const neonXorThreshold = 32

func exclusiveUnion2by2(set1 []uint16, set2 []uint16, buffer []uint16) int {
	if len(set1) < neonXorThreshold || len(set2) < neonXorThreshold || cap(buffer) < len(set1)+len(set2) {
		return localexclusiveUnion2by2(set1, set2, buffer)
	}
	// Callers such as xorArray pass a zero-length buffer with capacity.
	buffer = buffer[:cap(buffer)]
	outLen, pos1, pos2 := xorKernelNEON(set1, set2, buffer, &uniqshuf[0])
	// Everything left in either input is strictly greater than the last
	// emitted value, so the tails need no seam check.
	return outLen + localexclusiveUnion2by2(set1[pos1:], set2[pos2:], buffer[outLen:])
}

// uniqshuf[m] is the TBL index vector that compacts the lanes clear in mask m
// to the front; 0xFF entries zero the trailing lanes.
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
	return
}
