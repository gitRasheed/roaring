//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"math/rand"
	"testing"
)

func checkPair(t *testing.T, label string, a, b []uint16) {
	t.Helper()
	want := localintersect2by2Cardinality(a, b)
	got := intersection2by2Cardinality(a, b)
	if want != got {
		t.Fatalf("%s: la=%d lb=%d want %d got %d", label, len(a), len(b), want, got)
	}
	got = intersection2by2Cardinality(b, a)
	if want != got {
		t.Fatalf("%s swapped: la=%d lb=%d want %d got %d", label, len(b), len(a), want, got)
	}
	checkMaterialize(t, label, a, b)
	checkMaterialize(t, label+" swapped", b, a)
}

const canary = 0xABAB

// checkMaterialize runs intersection2by2 against the scalar reference under
// the two shipping buffer contracts: an exact min(la,lb)-capacity buffer
// with canaries beyond it (andArray), and a buffer aliasing set1 in place
// (iandArray).
func checkMaterialize(t *testing.T, label string, a, b []uint16) {
	t.Helper()
	m := len(a)
	if len(b) < m {
		m = len(b)
	}
	want := make([]uint16, m)
	wn := localintersect2by2(a, b, want)
	want = want[:wn]

	backing := make([]uint16, m+16)
	for i := range backing {
		backing[i] = canary
	}
	gn := intersection2by2(a, b, backing[0:0:m])
	if gn != wn {
		t.Fatalf("%s: la=%d lb=%d want len %d got %d", label, len(a), len(b), wn, gn)
	}
	for i := 0; i < wn; i++ {
		if backing[i] != want[i] {
			t.Fatalf("%s: la=%d lb=%d idx %d want %d got %d", label, len(a), len(b), i, want[i], backing[i])
		}
	}
	for i := m; i < m+16; i++ {
		if backing[i] != canary {
			t.Fatalf("%s: overstore past cap at %d (cap %d)", label, i, m)
		}
	}

	inplace := make([]uint16, len(a)+16)
	for i := range inplace {
		inplace[i] = canary
	}
	copy(inplace, a)
	gn = intersection2by2(inplace[:len(a)], b, inplace[0:len(a):len(a)])
	if gn != wn {
		t.Fatalf("%s inplace: la=%d lb=%d want len %d got %d", label, len(a), len(b), wn, gn)
	}
	for i := 0; i < wn; i++ {
		if inplace[i] != want[i] {
			t.Fatalf("%s inplace: idx %d want %d got %d", label, i, want[i], inplace[i])
		}
	}
	for i := len(a); i < len(a)+16; i++ {
		if inplace[i] != canary {
			t.Fatalf("%s inplace: overstore past cap at %d (cap %d)", label, i, len(a))
		}
	}
}

func TestIntersectNEONSmallSizes(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for la := 0; la <= 80; la++ {
		for lb := 60; lb <= 80; lb++ {
			a := genSortedUnique(rng, la, 160)
			b := genSortedUnique(rng, lb, 160)
			checkPair(t, "small", a, b)
		}
	}
}

func TestIntersectNEONIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for n := 64; n <= 1024; n *= 2 {
		a := genSortedUnique(rng, n, 4*n)
		checkPair(t, "identical", a, a)
	}
}

func TestIntersectNEONEvenOdd(t *testing.T) {
	for n := 64; n <= 2048; n *= 2 {
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			a[i] = uint16(2 * i)
			b[i] = uint16(2*i + 1)
		}
		checkPair(t, "evenodd", a, b)
	}
}

func TestIntersectNEONRangeDisjoint(t *testing.T) {
	// Fully separated ranges pin the dispatcher's O(1) pre-gate (the kernel
	// is never entered); the interleaved variant touches 0xFFFF inside the
	// kernel's fast-forward paths.
	for n := 64; n <= 512; n *= 2 {
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			a[i] = uint16(i)
			b[i] = uint16(65535 - n + 1 + i)
		}
		checkPair(t, "extremes", a, b)
		for i := 0; i < n; i++ {
			a[i] = uint16(65534 - 2*(n-1-i))
			b[i] = uint16(65535 - 2*(n-1-i))
		}
		checkPair(t, "extremes-interleaved", a, b)
	}
}

func TestIntersectNEONGallopingBoundary(t *testing.T) {
	// 32x2048 is just below the 64:1 galloping cutoff; 32x2049 crosses it.
	// Pin both, both orientations.
	rng := rand.New(rand.NewSource(5))
	small := genSortedUnique(rng, 32, 65536)
	for _, large := range []int{2048, 2049} {
		big := genSortedUnique(rng, large, 65536)
		checkPair(t, "gallop-boundary", small, big)
	}
}

func TestIntersectNEONInPlaceSpareCap(t *testing.T) {
	// iandArray passes set1's backing array with len==cardinality but often
	// cap>len; the wrapper reslices to cap, changing which store path runs.
	rng := rand.New(rand.NewSource(6))
	for _, n := range []int{64, 256, 1000} {
		a := genSortedUnique(rng, n, 4*n)
		b := genSortedUnique(rng, n, 4*n)
		want := make([]uint16, n)
		wn := localintersect2by2(a, b, want)
		backing := make([]uint16, n+64)
		for i := range backing {
			backing[i] = canary
		}
		copy(backing, a)
		gn := intersection2by2(backing[:n], b, backing[:n:n+40])
		if gn != wn {
			t.Fatalf("sparecap: n=%d want len %d got %d", n, wn, gn)
		}
		for i := 0; i < wn; i++ {
			if backing[i] != want[i] {
				t.Fatalf("sparecap: idx %d want %d got %d", i, want[i], backing[i])
			}
		}
		for i := n + 40; i < n+64; i++ {
			if backing[i] != canary {
				t.Fatalf("sparecap: overstore past cap at %d", i)
			}
		}
	}
}

func TestIntersectNEONEndpoint(t *testing.T) {
	// a's last equals b's first: the gate must not skip the boundary match.
	for n := 64; n <= 512; n *= 2 {
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			a[i] = uint16(i)
			b[i] = uint16(n - 1 + i)
		}
		checkPair(t, "endpoint", a, b)
	}
}

func TestIntersectNEONBlockRuns(t *testing.T) {
	// Alternating disjoint 8*run blocks: exercises fast-forward runs.
	for run := 1; run <= 32; run *= 2 {
		n := 2048
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			blk, off := i/(8*run), i%(8*run)
			a[i] = uint16(blk*16*run + off)
			b[i] = uint16(blk*16*run + 8*run + off)
		}
		checkPair(t, "blockruns", a, b)
	}
}

func TestIntersectNEONRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for iter := 0; iter < 500; iter++ {
		la := 64 + rng.Intn(4000)
		lb := 64 + rng.Intn(4000)
		if iter%5 == 0 {
			lb = 64 + rng.Intn(120) // skewed pairs
		}
		max := 256 + rng.Intn(65280)
		if la > max {
			la = max
		}
		if lb > max {
			lb = max
		}
		a := genSortedUnique(rng, la, max)
		b := genSortedUnique(rng, lb, max)
		checkPair(t, "random", a, b)
	}
}
