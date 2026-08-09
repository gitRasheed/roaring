//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"math/rand"
	"testing"
)

const canary = 0xABAB

// checkDifference verifies the exact-capacity, in-place and spare-capacity
// buffer contracts.
func checkDifference(t *testing.T, label string, a, b []uint16) {
	t.Helper()
	want := make([]uint16, len(a))
	wn := localdifference(a, b, want)
	want = want[:wn]

	m := len(a)
	backing := make([]uint16, m+16)
	for i := range backing {
		backing[i] = canary
	}
	gn := difference(a, b, backing[0:0:m])
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
	gn = difference(inplace[:len(a)], b, inplace[0:len(a):len(a)])
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

	spare := make([]uint16, len(a)+24)
	for i := range spare {
		spare[i] = canary
	}
	copy(spare, a)
	gn = difference(spare[:len(a)], b, spare[0:len(a):len(a)+8])
	if gn != wn {
		t.Fatalf("%s sparecap: la=%d lb=%d want len %d got %d", label, len(a), len(b), wn, gn)
	}
	for i := 0; i < wn; i++ {
		if spare[i] != want[i] {
			t.Fatalf("%s sparecap: idx %d want %d got %d", label, i, want[i], spare[i])
		}
	}
	for i := len(a) + 8; i < len(a)+24; i++ {
		if spare[i] != canary {
			t.Fatalf("%s sparecap: overstore past cap at %d (cap %d)", label, i, len(a)+8)
		}
	}
}

func checkDifferenceBoth(t *testing.T, label string, a, b []uint16) {
	t.Helper()
	checkDifference(t, label, a, b)
	checkDifference(t, label+" swapped", b, a)
}

func TestAndnotNEONSmallSizes(t *testing.T) {
	r := rand.New(rand.NewSource(41))
	for la := 0; la <= 40; la++ {
		for lb := 12; lb <= 44; lb++ {
			a := genSortedUnique(r, la, 160)
			b := genSortedUnique(r, lb, 160)
			checkDifferenceBoth(t, "small", a, b)
		}
	}
}

func TestAndnotNEONRangeDisjoint(t *testing.T) {
	for _, n := range []int{64, 512} {
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			a[i] = uint16(2 * i)
			b[i] = uint16(2*i + 1)
		}
		checkDifferenceBoth(t, "interleaved", a, b)

		for i := 0; i < n; i++ {
			a[i] = uint16(i)
			b[i] = uint16(65535 - n + 1 + i)
		}
		checkDifferenceBoth(t, "extremes", a, b)

		for i := 0; i < n; i++ {
			b[i] = uint16(n - 1 + i)
		}
		checkDifferenceBoth(t, "endpoint", a, b)
	}
}

// Skewed and sparse inputs exercise retained-block spill and drain handoffs.
func TestAndnotNEONSkewSpill(t *testing.T) {
	r := rand.New(rand.NewSource(43))
	small := genSortedUnique(r, 32, 60000)
	large := genSortedUnique(r, 2048, 60000)
	checkDifferenceBoth(t, "skew", small, large)

	a := make([]uint16, 512)
	for i := range a {
		a[i] = uint16(i * 3)
	}
	var b []uint16
	for i := 5; i < 512; i += 9 {
		b = append(b, uint16(i*3))
	}
	checkDifferenceBoth(t, "pickoff", a, b)
}

// Alternating disjoint runs exercise the fast-forward store/skip loop.
func TestAndnotNEONBlockRuns(t *testing.T) {
	for _, run := range []int{1, 2, 4, 8} {
		n := 1024
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			blk, off := i/(8*run), i%(8*run)
			a[i] = uint16(blk*16*run + off)
			b[i] = uint16(blk*16*run + 8*run + off)
		}
		checkDifferenceBoth(t, "blockruns", a, b)
	}
}

func TestAndnotNEONRandom(t *testing.T) {
	r := rand.New(rand.NewSource(47))
	for i := 0; i < 100; i++ {
		la := r.Intn(3000)
		lb := r.Intn(3000)
		valRange := 512 + r.Intn(65000)
		a := genSortedUnique(r, la, valRange)
		b := genSortedUnique(r, lb, valRange)
		checkDifferenceBoth(t, "random", a, b)
	}
}

// Three-way aliasing must return empty without writing past the capacity.
func TestAndnotNEONSelfAlias(t *testing.T) {
	r := rand.New(rand.NewSource(53))
	for _, n := range []int{16, 65, 512} {
		a := genSortedUnique(r, n, 4*n)
		self := make([]uint16, n+16)
		for i := range self {
			self[i] = canary
		}
		copy(self, a)
		gn := difference(self[:n], self[:n], self[0:n:n])
		if gn != 0 {
			t.Fatalf("self n=%d: want empty, got len %d", n, gn)
		}
		for i := n; i < n+16; i++ {
			if self[i] != canary {
				t.Fatalf("self n=%d: overstore past cap at %d", n, i)
			}
		}
		checkDifference(t, "equal-distinct", a, append([]uint16(nil), a...))
	}

	rb := BitmapOf()
	for i := 0; i < 500; i++ {
		rb.Add(uint32(i * 5))
	}
	rb.AndNot(rb)
	if !rb.IsEmpty() {
		t.Fatalf("bitmap self AndNot: want empty, got %d", rb.GetCardinality())
	}
}

// Duplicate-bearing inputs pass Validate: results unspecified, stores bounded.
func TestAndnotNEONDuplicateInputBounded(t *testing.T) {
	dup := func(vals ...uint16) []uint16 { return vals }
	cases := [][2][]uint16{
		{dup(1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
			dup(1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34)},
		{dup(100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 103, 103),
			dup(100, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102)},
	}
	for ci, c := range cases {
		for o := 0; o < 2; o++ {
			a, b := c[0], c[1]
			if o == 1 {
				a, b = b, a
			}
			m := len(a)
			backing := make([]uint16, m+16)
			for i := range backing {
				backing[i] = canary
			}
			gn := difference(a, b, backing[0:0:m])
			if gn < 0 || gn > m {
				t.Fatalf("dup case %d/%d: out of range length %d", ci, o, gn)
			}
			for i := m; i < m+16; i++ {
				if backing[i] != canary {
					t.Fatalf("dup case %d/%d: overstore past cap at %d", ci, o, i)
				}
			}
			inplace := make([]uint16, m+16)
			for i := range inplace {
				inplace[i] = canary
			}
			copy(inplace, a)
			gn = difference(inplace[:m], b, inplace[0:m:m])
			if gn < 0 || gn > m {
				t.Fatalf("dup case %d/%d inplace: out of range length %d", ci, o, gn)
			}
			for i := m; i < m+16; i++ {
				if inplace[i] != canary {
					t.Fatalf("dup case %d/%d inplace: overstore past cap at %d", ci, o, i)
				}
			}
		}
	}
}

// Kernel exits must report exact cursor, spill block and mask state.
func TestAndnotNEONKernelHandoffs(t *testing.T) {
	a := []uint16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	buffer := make([]uint16, len(a))
	var spill [8]uint16
	out, pos1, pos2, spilled, mask := andnotKernelNEON(a, a, buffer, &uniqshuf[0], &spill)
	if out != 0 || pos1 != 16 || pos2 != 16 || spilled != 0 || mask != 0 {
		t.Fatalf("tie exit: out=%d pos1=%d pos2=%d spilled=%d mask=%08b", out, pos1, pos2, spilled, mask)
	}

	sa := []uint16{100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 204, 205, 206, 207}
	sb := []uint16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 103}
	out, pos1, pos2, spilled, mask = andnotKernelNEON(sa, sb, buffer, &uniqshuf[0], &spill)
	wantSpill := [8]uint16{100, 101, 102, 103, 104, 105, 106, 107}
	if out != 0 || pos1 != 8 || pos2 != 16 || spilled != 1 || mask != 1<<3 || spill != wantSpill {
		t.Fatalf("spill: out=%d pos1=%d pos2=%d spilled=%d mask=%08b spill=%v", out, pos1, pos2, spilled, mask, spill)
	}
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
