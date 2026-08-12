//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"math/rand"
	"reflect"
	"testing"
)

func checkPart(t *testing.T, a, b []uint16, label string) {
	t.Helper()
	want := refUnion(a, b)
	buffer := make([]uint16, 0, len(a)+len(b))
	n := unionNEONPart(a, b, buffer)
	got := buffer[:n]
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("%s: la=%d lb=%d: want %d elems, got %d\nwant %v\ngot  %v",
			label, len(a), len(b), len(want), n, want, got)
	}
}

func TestPartRandom(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for iter := 0; iter < 3000; iter++ {
		valRange := 256 + r.Intn(65280)
		a := genSortedUnique(r, r.Intn(4096), valRange)
		b := genSortedUnique(r, r.Intn(4096), valRange)
		checkPart(t, a, b, "random")
	}
}

// Sizes around the 48-element reserve and the 16-element window, where the
// vector loop runs zero, one, or a partial number of iterations.
func TestPartSizeMatrix(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 47, 48, 49, 63, 64, 79, 80, 81, 95, 96, 97, 112, 127, 128, 129, 255, 256}
	for _, la := range sizes {
		for _, lb := range sizes {
			a := make([]uint16, la)
			b := make([]uint16, lb)
			for i := range a {
				a[i] = uint16(2 * i)
			}
			for i := range b {
				b[i] = uint16(3*i + 1)
			}
			checkPart(t, a, b, "sizes")
		}
	}
}

// Every window seam shares its boundary value, so amax == bmax and T = 32
// with a duplicate in the last lane pair.
func TestPartEqualMaxima(t *testing.T) {
	for windows := 2; windows <= 32; windows *= 2 {
		var a, b []uint16
		next := uint16(0)
		for i := 0; i < windows; i++ {
			for l := 0; l < 16; l++ {
				a = append(a, next+uint16(l))
			}
			next += 15
			for l := 0; l < 16; l++ {
				b = append(b, next+uint16(l))
			}
			next += 15
		}
		checkPart(t, a, b, "equal-maxima")
		checkPart(t, b, a, "equal-maxima-swap")
	}
}

// Identical inputs: every merged value appears twice, T is 32 every
// iteration, and half of every vector is dropped by the dedup.
func TestPartIdentical(t *testing.T) {
	for _, n := range []int{80, 96, 128, 257, 1024} {
		a := make([]uint16, n)
		for i := range a {
			a[i] = uint16(3 * i)
		}
		checkPart(t, a, a, "identical")
	}
}

// One window entirely below the other: T = 16, and the whole upper half of
// the merge is clamped away.
func TestPartDisjointRuns(t *testing.T) {
	for _, run := range []int{16, 17, 31, 32, 64} {
		var a, b []uint16
		v := 0
		for len(a) < 600 || len(b) < 600 {
			for i := 0; i < run; i++ {
				a = append(a, uint16(v))
				v++
			}
			for i := 0; i < run; i++ {
				b = append(b, uint16(v))
				v++
			}
		}
		checkPart(t, a, b, "disjoint-runs")
		checkPart(t, b, a, "disjoint-runs-swap")
	}
}

// 0xFFFF in the last lanes of both inputs, and blocks that reach it from
// several distances, exercising the all-ones seam carry initialization.
func TestPartHighEnd(t *testing.T) {
	span := func(lo, hi int) []uint16 {
		var s []uint16
		for v := lo; v <= hi; v++ {
			s = append(s, uint16(v))
		}
		return s
	}
	a := span(65536-200, 65535)
	b := span(65536-201, 65535)
	checkPart(t, a, b, "high-overlap")
	checkPart(t, b, a, "high-overlap-swap")

	c := append(span(0, 99), span(65536-100, 65535)...)
	d := append(span(0, 98), span(65536-101, 65535)...)
	checkPart(t, c, d, "high-split")
	checkPart(t, d, c, "high-split-swap")

	e := span(65536-100, 65535)
	checkPart(t, e, e, "high-identical")
}

// Duplicates that land on a vector boundary inside the merged 32, so the
// dedup compare has to reach across from the previous vector's lane 7.
func TestPartVectorSeamDuplicates(t *testing.T) {
	for off := 0; off < 32; off++ {
		a := make([]uint16, 200)
		b := make([]uint16, 200)
		for i := range a {
			a[i] = uint16(4 * i)
		}
		for i := range b {
			b[i] = uint16(4*i + 2)
		}
		// plant a shared value at a shifting position
		b[off%len(b)] = a[(off*3)%len(a)]
		sortU16(b)
		if !strictlySorted(b) {
			continue
		}
		checkPart(t, a, b, "seam-dup")
	}
}

func sortU16(s []uint16) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func strictlySorted(s []uint16) bool {
	for i := 1; i < len(s); i++ {
		if s[i] <= s[i-1] {
			return false
		}
	}
	return true
}

// The stores write 16 elements at a time; nothing may touch beyond
// len(set1)+len(set2).
func TestPartBufferCanaries(t *testing.T) {
	r := rand.New(rand.NewSource(4242))
	for iter := 0; iter < 400; iter++ {
		a := genSortedUnique(r, 80+r.Intn(1200), 8192)
		b := genSortedUnique(r, 80+r.Intn(1200), 8192)
		need := len(a) + len(b)
		full := make([]uint16, need+32)
		for i := need; i < len(full); i++ {
			full[i] = 0xDEAD
		}
		n := unionNEONPart(a, b, full[0:0:need])
		if !reflect.DeepEqual(refUnion(a, b), full[:n]) {
			t.Fatalf("canary iter %d: wrong result", iter)
		}
		for i := need; i < len(full); i++ {
			if full[i] != 0xDEAD {
				t.Fatalf("canary iter %d: guard word %d clobbered (0x%04X)", iter, i, full[i])
			}
		}
	}
}

// iorArray's geometry: the output buffer starts len(set2) elements below
// set1 in the same backing array, so stores overwrite set1 behind the read
// cursor. The kernel's reserve is what keeps the writes behind the reads.
func TestPartAliasGeometry(t *testing.T) {
	r := rand.New(rand.NewSource(555))
	for _, l2 := range []int{80, 81, 96, 128, 512, 2048} {
		for iter := 0; iter < 40; iter++ {
			l1 := 80 + r.Intn(3000)
			set1 := genSortedUnique(r, l1, 65536)
			set2 := genSortedUnique(r, l2, 65536)
			l1, l2 = len(set1), len(set2)
			want := refUnion(set1, set2)
			shared := make([]uint16, l1+l2)
			copy(shared[l2:], set1)
			n := unionNEONPart(shared[l2:], set2, shared)
			if !reflect.DeepEqual(want, shared[:n]) {
				t.Fatalf("alias l1=%d l2=%d: want %d elems got %d", l1, l2, len(want), n)
			}
		}
	}
}

// Densest alias geometry: set1 and set2 fully interleaved, which keeps the
// output cursor as close to the read cursors as the algorithm allows.
func TestPartAliasInterleaved(t *testing.T) {
	for _, n := range []int{80, 128, 1024, 4096} {
		set1 := make([]uint16, n)
		set2 := make([]uint16, n)
		for i := 0; i < n; i++ {
			set1[i] = uint16(2 * i)
			set2[i] = uint16(2*i + 1)
		}
		want := refUnion(set1, set2)
		shared := make([]uint16, 2*n)
		copy(shared[n:], set1)
		got := shared[:unionNEONPart(shared[n:], set2, shared)]
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("alias interleaved n=%d: want %d got %d", n, len(want), len(got))
		}
	}
}

// lazyorArray passes a zero-length buffer with spare capacity.
func TestPartZeroLenBuffer(t *testing.T) {
	r := rand.New(rand.NewSource(1001))
	for iter := 0; iter < 200; iter++ {
		a := genSortedUnique(r, 80+r.Intn(900), 4096)
		b := genSortedUnique(r, 80+r.Intn(900), 4096)
		want := refUnion(a, b)
		buf := make([]uint16, 0, len(a)+len(b))
		got := buf[:unionNEONPart(a, b, buf)]
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("zero-len buffer iter %d", iter)
		}
	}
}

// The in-place self-union geometry the alias guard exists for: buffer and
// set2 share a base pointer. Without the guard the stores overrun set2.
func TestPartSelfUnionGuard(t *testing.T) {
	n := 512
	content := make([]uint16, 2*n)
	set2 := content[:n]
	for i := range set2 {
		set2[i] = uint16(3 * i)
	}
	set1 := content[n:]
	copy(set1, set2)
	want := refUnion(set1, set2)
	got := content[:unionNEONPart(set1, set2, content)]
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("self-union: want %d elems got %d", len(want), len(got))
	}
}

// Cross-check the kernel's own accounting: the partition consumes at least
// 16 and at most 32 elements per iteration, and every element left behind is
// strictly greater than the last one emitted.
func TestPartKernelInvariants(t *testing.T) {
	r := rand.New(rand.NewSource(2024))
	for iter := 0; iter < 500; iter++ {
		a := genSortedUnique(r, 80+r.Intn(2000), 65536)
		b := genSortedUnique(r, 80+r.Intn(2000), 65536)
		buf := make([]uint16, len(a)+len(b)+32)
		outLen, pos1, pos2 := unionPartKernelNEON(a, b, buf, &uniqshuf[0])
		if outLen == 0 {
			continue
		}
		last := buf[outLen-1]
		if pos1 < len(a) && a[pos1] <= last {
			t.Fatalf("iter %d: set1 leftover %d <= last emitted %d", iter, a[pos1], last)
		}
		if pos2 < len(b) && b[pos2] <= last {
			t.Fatalf("iter %d: set2 leftover %d <= last emitted %d", iter, b[pos2], last)
		}
		want := refUnion(a[:pos1], b[:pos2])
		if !reflect.DeepEqual(want, buf[:outLen]) {
			t.Fatalf("iter %d: kernel output is not the union of what it consumed", iter)
		}
		// the reserve must leave room for the 32-element stores
		if len(a)-pos1 < 8 && len(b)-pos2 < 8 {
			t.Fatalf("iter %d: reserve exhausted (%d, %d left)", iter, len(a)-pos1, len(b)-pos2)
		}
	}
}

// The window fast path must use a strict comparison: when set1's window
// maximum equals set2's window minimum the value has to reach the dedup, so
// these inputs die if the branch is taken on equality.
func TestPartFastPathTie(t *testing.T) {
	for _, run := range []int{16, 32, 48} {
		var a, b []uint16
		v := 0
		for len(a) < 800 || len(b) < 800 {
			for i := 0; i < run; i++ {
				a = append(a, uint16(v))
				v++
			}
			v-- // b's run starts on a's last value
			for i := 0; i < run; i++ {
				b = append(b, uint16(v))
				v++
			}
			v-- // and a's next run starts on b's last value
		}
		checkPart(t, a, b, "fastpath-tie")
		checkPart(t, b, a, "fastpath-tie-swap")
	}
}

// The benchmark shapes are load-bearing evidence, so they get the same
// correctness check as any other input, plus the lengths and strict sorting
// the measurements assume.
func TestPartBenchShapes(t *testing.T) {
	shapes := []string{"interleaved", "shared7of8", "shared95", "random", "disjointcoin",
		"runs16", "dense50", "spread", "runs4", "runs8", "runs32",
		"tileshuf", "tileper", "minprogress", "identical"}
	for _, shape := range shapes {
		for v := 0; v < 4; v++ {
			a, b := partPairSeed(shape, 4096, int64(v))
			if !strictlySorted(a) || !strictlySorted(b) {
				t.Fatalf("%s variant %d: input not strictly sorted", shape, v)
			}
			if len(a) != 4096 || len(b) != 4096 {
				t.Fatalf("%s variant %d: lengths %d, %d", shape, v, len(a), len(b))
			}
			checkPart(t, a, b, shape)
		}
	}
}

// Ownership of each 16-element window drawn at random, so the fast path and
// the general path interleave unpredictably.
func TestPartMixedOwnership(t *testing.T) {
	r := rand.New(rand.NewSource(8080))
	for iter := 0; iter < 200; iter++ {
		var a, b []uint16
		v := 0
		for len(a) < 700 || len(b) < 700 {
			run := 1 + r.Intn(40)
			own := r.Intn(2) == 0
			for i := 0; i < run; i++ {
				if own {
					a = append(a, uint16(v))
				} else {
					b = append(b, uint16(v))
				}
				if r.Intn(8) == 0 { // occasionally share the value
					if own {
						b = append(b, uint16(v))
					} else {
						a = append(a, uint16(v))
					}
				}
				v++
			}
		}
		checkPart(t, a, b, "mixed-ownership")
	}
}
