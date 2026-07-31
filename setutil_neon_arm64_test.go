//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"math/rand"
	"reflect"
	"testing"
)

func refUnion(a, b []uint16) []uint16 {
	out := make([]uint16, len(a)+len(b))
	n := scalarMergeUnion(a, b, out)
	return out[:n]
}

func genSortedUnique(r *rand.Rand, n, valRange int) []uint16 {
	if n > valRange {
		n = valRange
	}
	seen := make(map[uint16]bool, n)
	for len(seen) < n {
		seen[uint16(r.Intn(valRange))] = true
	}
	out := make([]uint16, 0, n)
	for v := 0; v < valRange; v++ {
		if seen[uint16(v)] {
			out = append(out, uint16(v))
		}
	}
	return out
}

func checkUnion(t *testing.T, a, b []uint16, label string) {
	t.Helper()
	want := refUnion(a, b)
	buffer := make([]uint16, len(a)+len(b))
	got := buffer[:union2by2(a, b, buffer)]
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("%s: len(a)=%d len(b)=%d: want %d elems, got %d\nwant %v\ngot  %v",
			label, len(a), len(b), len(want), len(got), want, got)
	}
}

func TestUnion2By2NEONSmallSizes(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for la := 0; la <= 40; la++ {
		for lb := 0; lb <= 40; lb++ {
			a := genSortedUnique(r, la, 64)
			b := genSortedUnique(r, lb, 64)
			checkUnion(t, a, b, "small-dense")
		}
	}
}

func TestUnion2By2NEONAdversarial(t *testing.T) {
	for n := 8; n <= 2048; n *= 2 {
		identical := make([]uint16, n)
		evens := make([]uint16, n)
		odds := make([]uint16, n)
		low := make([]uint16, n)
		high := make([]uint16, n)
		for i := 0; i < n; i++ {
			identical[i] = uint16(3 * i)
			evens[i] = uint16(2 * i)
			odds[i] = uint16(2*i + 1)
			low[i] = uint16(i)
			high[i] = uint16(65535 - n + 1 + i)
		}
		checkUnion(t, identical, identical, "identical")
		checkUnion(t, evens, odds, "interleaved")
		checkUnion(t, odds, evens, "interleaved-swap")
		checkUnion(t, low, high, "disjoint-extremes")
		checkUnion(t, high, low, "disjoint-extremes-swap")
	}
}

func TestUnion2By2NEONRandom(t *testing.T) {
	r := rand.New(rand.NewSource(12345))
	for iter := 0; iter < 2000; iter++ {
		valRange := 256 + r.Intn(65280)
		a := genSortedUnique(r, r.Intn(5000), valRange)
		b := genSortedUnique(r, r.Intn(5000), valRange)
		checkUnion(t, a, b, "random")
	}
}

// forceNEON runs f with the kernel engaged regardless of the dispatch
// threshold, by calling the kernel path pieces the way union2by2 does.
func unionViaKernel(set1, set2, buffer []uint16) int {
	buffer = buffer[:cap(buffer)]
	var leftover [16]uint16
	outLen, pos1, pos2, ll := unionKernelNEON(set1, set2, buffer, &uniqshuf[0], &leftover)
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

func checkKernel(t *testing.T, a, b []uint16, label string) {
	t.Helper()
	if len(a) < 8 || len(b) < 8 {
		checkUnion(t, a, b, label)
		return
	}
	want := refUnion(a, b)
	buffer := make([]uint16, 0, len(a)+len(b))
	n := unionViaKernel(a, b, buffer)
	got := buffer[:n]
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("%s: la=%d lb=%d want %v got %v", label, len(a), len(b), want, got)
	}
}

// Codex-review hardening: dispatch/block boundary matrix with zero-len
// buffers, both input orders.
func TestUnion2By2NEONBoundaryMatrix(t *testing.T) {
	for _, sz := range [][2]int{{1, 1}, {2, 8}, {7, 8}, {8, 8}, {8, 9}, {8, 16}, {15, 16}, {16, 17}} {
		a := make([]uint16, sz[0])
		b := make([]uint16, sz[1])
		for i := range a {
			a[i] = uint16(2 * i)
		}
		for i := range b {
			b[i] = uint16(3*i + 1)
		}
		checkKernel(t, a, b, "boundary")
		checkKernel(t, b, a, "boundary-swap")
	}
}

// A duplicate (7) straddling lanes 7/0 across consecutive stores, plus
// identical high-end blocks touching 0xFFFF in the carry flush.
func TestUnion2By2NEONLaneStraddleAndHighEnd(t *testing.T) {
	a := []uint16{0, 2, 4, 6, 7, 20, 22, 24}
	b := []uint16{1, 3, 5, 7, 21, 23, 25, 27}
	checkKernel(t, a, b, "straddle")
	checkKernel(t, b, a, "straddle-swap")

	high := make([]uint16, 8)
	for i := range high {
		high[i] = uint16(65528 + i)
	}
	checkKernel(t, high, high, "identical-high")
	low := make([]uint16, 8)
	for i := range low {
		low[i] = uint16(65520 + i)
	}
	checkKernel(t, low, high, "adjacent-high")
}

// Tightest valid alias geometry: set1 begins at output offset 8 (= len(set2))
// and set2 has exactly one block, forcing set2 exhaustion and the
// minimum-gap lookahead tail path.
func TestUnion2By2NEONMinimumGapAlias(t *testing.T) {
	set2 := []uint16{0, 1, 2, 3, 4, 5, 6, 7}
	set1 := make([]uint16, 64)
	for i := range set1 {
		set1[i] = uint16(100 + i)
	}
	want := refUnion(set1, set2)
	shared := make([]uint16, len(set1)+len(set2))
	copy(shared[len(set2):], set1)
	n := unionViaKernel(shared[len(set2):], set2, shared)
	if !reflect.DeepEqual(want, shared[:n]) {
		t.Fatalf("minimum-gap alias: want %d elems got %d", len(want), n)
	}
}

// Guard region beyond len(set1)+len(set2) must never be touched by the
// kernel's fixed-width 16-byte stores.
func TestUnion2By2NEONBufferCanaries(t *testing.T) {
	r := rand.New(rand.NewSource(31337))
	for iter := 0; iter < 300; iter++ {
		a := genSortedUnique(r, 8+r.Intn(1000), 8192)
		b := genSortedUnique(r, 8+r.Intn(1000), 8192)
		need := len(a) + len(b)
		full := make([]uint16, need+16)
		for i := need; i < len(full); i++ {
			full[i] = 0xDEAD
		}
		n := unionViaKernel(a, b, full[0:0:need])
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

// Reproduces the lazyorArray pattern: buffer arrives with len 0 and spare
// capacity; union2by2 must use the capacity (historical asm contract).
func TestUnion2By2NEONZeroLenBuffer(t *testing.T) {
	r := rand.New(rand.NewSource(99))
	for iter := 0; iter < 200; iter++ {
		a := genSortedUnique(r, 8+r.Intn(500), 4096)
		b := genSortedUnique(r, 8+r.Intn(500), 4096)
		want := refUnion(a, b)
		buffer := make([]uint16, 0, len(a)+len(b))
		n := union2by2(a, b, buffer)
		got := buffer[:n]
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("zero-len buffer: mismatch (want %d got %d elems)", len(want), n)
		}
	}
}

// Reproduces the iorArray aliasing pattern (arraycontainer.go): set1 lives in
// the upper region of the same array the output is written into. The vector
// path's 16-byte stores must never clobber unread set1 data.
func TestUnion2By2NEONAliasedBuffer(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for iter := 0; iter < 500; iter++ {
		valRange := 256 + r.Intn(65280)
		set1 := genSortedUnique(r, 8+r.Intn(2000), valRange)
		set2 := genSortedUnique(r, 8+r.Intn(2000), valRange)
		want := refUnion(set1, set2)

		max := len(set1) + len(set2)
		shared := make([]uint16, max)
		copy(shared[len(set2):max], set1)
		got := shared[:union2by2(shared[len(set2):max], set2, shared)]
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("aliased: len1=%d len2=%d: mismatch (want %d got %d elems)",
				len(set1), len(set2), len(want), len(got))
		}
	}
}
