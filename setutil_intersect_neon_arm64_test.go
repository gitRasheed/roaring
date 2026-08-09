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
	for la := 0; la <= 40; la++ {
		for lb := 12; lb <= 44; lb++ {
			a := genSortedUnique(rng, la, 120)
			b := genSortedUnique(rng, lb, 120)
			checkPair(t, "small", a, b)
		}
	}
}

func TestIntersectNEONRangeDisjoint(t *testing.T) {
	// The interleaved variant touches 0xFFFF inside the kernel's
	// fast-forward paths; the endpoint variant pins the boundary match the
	// range gate must not skip.
	for _, n := range []int{64, 512} {
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			a[i] = uint16(65534 - 2*(n-1-i))
			b[i] = uint16(65535 - 2*(n-1-i))
		}
		checkPair(t, "extremes-interleaved", a, b)
		for i := 0; i < n; i++ {
			a[i] = uint16(i)
			b[i] = uint16(n - 1 + i)
		}
		checkPair(t, "endpoint", a, b)
	}
}

func TestIntersectNEONGallopingBoundary(t *testing.T) {
	// 32x2048 stays on the kernel; 32x2049 crosses the 64:1 cutoff into
	// galloping. Pin both, both orientations.
	rng := rand.New(rand.NewSource(5))
	small := genSortedUnique(rng, 32, 65536)
	for _, large := range []int{2048, 2049} {
		big := genSortedUnique(rng, large, 65536)
		checkPair(t, "gallop-boundary", small, big)
	}
}

func TestIntersectNEONRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for iter := 0; iter < 100; iter++ {
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

// iandArray's self-intersection geometry (And of a bitmap with itself):
// set1, set2, and the output are the same backing array.
func TestIntersectNEONSelfAlias(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for iter := 0; iter < 25; iter++ {
		n := 8 + rng.Intn(3000)
		set := genSortedUnique(rng, n, 256+rng.Intn(65280))
		n = len(set)

		shared := make([]uint16, n, n+16)
		copy(shared, set)
		got := shared[:intersection2by2(shared, shared, shared)]
		if len(got) != len(set) {
			t.Fatalf("self-alias: n=%d got len %d", n, len(got))
		}
		for i := range set {
			if got[i] != set[i] {
				t.Fatalf("self-alias: n=%d idx %d want %d got %d", n, i, set[i], got[i])
			}
		}

		// set1 == set2 with an independent buffer (top-level And of a
		// bitmap with itself).
		out := make([]uint16, n)
		if g := intersection2by2(set, set, out[:0:n]); g != len(set) {
			t.Fatalf("same-inputs: n=%d want %d got %d", n, len(set), g)
		}
	}

	// iandArray with spare capacity: len==cardinality but cap>len changes
	// which store path runs.
	a := genSortedUnique(rng, 64, 256)
	b := genSortedUnique(rng, 64, 256)
	want := make([]uint16, 64)
	wn := localintersect2by2(a, b, want)
	backing := make([]uint16, 64+64)
	for i := range backing {
		backing[i] = canary
	}
	copy(backing, a)
	gn := intersection2by2(backing[:64], b, backing[:0:64+40])
	if gn != wn {
		t.Fatalf("sparecap: want len %d got %d", wn, gn)
	}
	for i := 0; i < wn; i++ {
		if backing[i] != want[i] {
			t.Fatalf("sparecap: idx %d want %d got %d", i, want[i], backing[i])
		}
	}
	for i := 64 + 40; i < 64+64; i++ {
		if backing[i] != canary {
			t.Fatalf("sparecap: overstore past cap at %d", i)
		}
	}
}

// Results on duplicate-bearing input are unspecified; the pinned guarantee
// is that no store escapes the buffer's capacity.
func TestIntersectNEONDuplicateInputBounded(t *testing.T) {
	check := func(label string, a, b []uint16) {
		t.Helper()
		m := len(a)
		if len(b) < m {
			m = len(b)
		}
		backing := make([]uint16, m+16)
		for i := range backing {
			backing[i] = canary
		}
		func() {
			// Unspecified results may surface as slice-bounds panics in
			// the scalar tail; only out-of-bounds stores are failures.
			defer func() { _ = recover() }()
			intersection2by2(a, b, backing[0:0:m])
			intersection2by2Cardinality(a, b)
		}()
		for i := m; i < m+16; i++ {
			if backing[i] != canary {
				t.Fatalf("%s: store past cap at index %d", label, i)
			}
		}
	}

	flat := make([]uint16, 64)
	for i := range flat {
		flat[i] = 100
	}
	ramp := make([]uint16, 32)
	for i := range ramp {
		ramp[i] = uint16(100 + i)
	}
	seam := make([]uint16, 33)
	for i := 0; i < 32; i++ {
		seam[i] = uint16(69 + i)
	}
	seam[32] = 100
	check("flat", flat, ramp)
	check("flat-swapped", ramp, flat)
	check("dup-seam", ramp, seam)
	check("dup-seam-swapped", seam, ramp)
}

// Slice starts misaligned independently for set1, set2, and the output:
// NEON loads and stores must be correct across 16-byte address boundaries.
func TestIntersectNEONMisalignment(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for _, n := range []int{16, 17, 24, 25, 64, 65} {
		for o1 := 0; o1 < 8; o1++ {
			for o2 := 0; o2 < 8; o2++ {
				for oo := 0; oo < 8; oo++ {
					a := genSortedUnique(rng, n, 3*n)
					b := genSortedUnique(rng, n, 3*n)
					back1 := make([]uint16, o1+len(a)+8)
					back2 := make([]uint16, o2+len(b)+8)
					copy(back1[o1:], a)
					copy(back2[o2:], b)
					s1 := back1[o1 : o1+len(a)]
					s2 := back2[o2 : o2+len(b)]

					m := len(a)
					if len(b) < m {
						m = len(b)
					}
					want := make([]uint16, m)
					wn := localintersect2by2(a, b, want)

					for _, pair := range [][2][]uint16{{s1, s2}, {s2, s1}} {
						backo := make([]uint16, oo+m+16)
						for i := range backo {
							backo[i] = canary
						}
						out := backo[oo : oo : oo+m]
						gn := intersection2by2(pair[0], pair[1], out)
						if gn != wn {
							t.Fatalf("n=%d o1=%d o2=%d oo=%d: len want %d got %d", n, o1, o2, oo, wn, gn)
						}
						for i := 0; i < wn; i++ {
							if backo[oo+i] != want[i] {
								t.Fatalf("n=%d o1=%d o2=%d oo=%d idx %d: want %d got %d", n, o1, o2, oo, i, want[i], backo[oo+i])
							}
						}
						for i := oo + m; i < len(backo); i++ {
							if backo[i] != canary {
								t.Fatalf("n=%d o1=%d o2=%d oo=%d: overstore at %d", n, o1, o2, oo, i)
							}
						}
						if c := intersection2by2Cardinality(pair[0], pair[1]); c != wn {
							t.Fatalf("n=%d o1=%d o2=%d card: want %d got %d", n, o1, o2, wn, c)
						}
					}
				}
			}
		}
	}
}
