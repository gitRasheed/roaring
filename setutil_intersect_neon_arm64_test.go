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
