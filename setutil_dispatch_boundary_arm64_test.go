//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"fmt"
	"math/rand"
	"testing"
)

// The dispatch gate moved from 256 to 64: these tests pin the public
// union2by2 path and the real caller geometries on both sides of the new
// boundary and of the old one.

func boundaryPair(r *rand.Rand, shape string, n1, n2 int) (a, b []uint16) {
	switch shape {
	case "sparse":
		return genSortedUnique(r, n1, 65536), genSortedUnique(r, n2, 65536)
	case "dense":
		vr := 3 * (n1 + n2) / 2
		return genSortedUnique(r, n1, vr), genSortedUnique(r, n2, vr)
	case "sharedprefix":
		// heavy overlap: a takes the pool's head, b its tail, sharing all
		// but ~1/8 of the smaller side
		small := n1
		if n2 < small {
			small = n2
		}
		big := n1 + n2 - small
		pool := genSortedUnique(r, big+small/8+1, 65536)
		return append([]uint16(nil), pool[:n1]...),
			append([]uint16(nil), pool[len(pool)-n2:]...)
	case "runs8":
		a = make([]uint16, 0, n1)
		b = make([]uint16, 0, n2)
		v := 0
		for len(a) < n1 || len(b) < n2 {
			for i := 0; i < 8 && len(a) < n1; i++ {
				a = append(a, uint16(v))
				v++
			}
			for i := 0; i < 8 && len(b) < n2; i++ {
				b = append(b, uint16(v))
				v++
			}
		}
		return a, b
	}
	panic("unknown shape")
}

func TestUnion2By2DispatchBoundary(t *testing.T) {
	pairs := [][2]int{
		{63, 63}, {63, 64}, {64, 63}, {64, 64}, {64, 65}, {65, 64},
		{63, 4096}, {64, 4096}, {4096, 64}, {65, 255}, {255, 65},
		{255, 255}, {255, 256}, {256, 255}, {256, 256},
	}
	for _, shape := range []string{"sparse", "dense", "sharedprefix", "runs8"} {
		for _, p := range pairs {
			for seed := int64(0); seed < 4; seed++ {
				r := rand.New(rand.NewSource(9000 + seed))
				a, b := boundaryPair(r, shape, p[0], p[1])
				want := refUnion(a, b)
				buffer := make([]uint16, len(a)+len(b))
				got := buffer[:union2by2(a, b, buffer)]
				label := fmt.Sprintf("%s/%dx%d/s%d", shape, p[0], p[1], seed)
				if len(got) != len(want) {
					t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
				}
				for k := range want {
					if got[k] != want[k] {
						t.Fatalf("%s: [%d]=%d, want %d", label, k, got[k], want[k])
					}
				}
			}
		}
	}
}

// TestArrayContainerUnionBoundary drives the three production caller
// geometries — iorArray's relocated in-place alias, self-union, orArray and
// lazyorArray's fresh-buffer calls — across the dispatch boundary.
func TestArrayContainerUnionBoundary(t *testing.T) {
	sizes := []int{63, 64, 65, 128, 255, 256, 300}
	for _, shape := range []string{"sparse", "dense", "sharedprefix", "runs8"} {
		for _, n1 := range sizes {
			for _, n2 := range sizes {
				r := rand.New(rand.NewSource(int64(31*n1 + n2)))
				av, bv := boundaryPair(r, shape, n1, n2)
				want := refUnion(av, bv)
				label := fmt.Sprintf("%s/%dx%d", shape, n1, n2)

				check := func(kind string, c container) {
					t.Helper()
					got := c.(*arrayContainer).content
					if len(got) != len(want) {
						t.Fatalf("%s/%s: length %d, want %d", label, kind, len(got), len(want))
					}
					for k := range want {
						if got[k] != want[k] {
							t.Fatalf("%s/%s: [%d]=%d, want %d", label, kind, k, got[k], want[k])
						}
					}
				}

				mk := func(v []uint16) *arrayContainer {
					return &arrayContainer{content: append([]uint16(nil), v...)}
				}

				check("orArray", mk(av).orArray(mk(bv)))
				check("lazyorArray", mk(av).lazyorArray(mk(bv)))
				check("iorArray", mk(av).iorArray(mk(bv)))
				// iorArray with pre-grown capacity: relocation stays in place
				acap := &arrayContainer{content: append(make([]uint16, 0, 2*(n1+n2)), av...)}
				check("iorArrayCap", acap.iorArray(mk(bv)))
				// self-union, both capacity regimes: the pre-grown one keeps
				// set2 and buffer sharing a backing array from offset 0 (the
				// wrapper's guarded geometry); the exact-cap one relocates.
				for kind, self := range map[string]*arrayContainer{
					"selfIor":    mk(av),
					"selfIorCap": {content: append(make([]uint16, 0, 2*n1), av...)},
				} {
					selfGot := self.iorArray(self).(*arrayContainer).content
					if len(selfGot) != n1 {
						t.Fatalf("%s/%s: length %d, want %d", label, kind, len(selfGot), n1)
					}
					for k, v := range av {
						if selfGot[k] != v {
							t.Fatalf("%s/%s: [%d]=%d, want %d", label, kind, k, selfGot[k], v)
						}
					}
				}
			}
		}
	}
}
