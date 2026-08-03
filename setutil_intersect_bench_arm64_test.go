//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"fmt"
	"math/rand"
	"testing"
)

func benchIntersectPair(shape string, n int, seed int64) (a, b []uint16) {
	r := rand.New(rand.NewSource(int64(42+n) + seed*7919))
	switch shape {
	case "dense50", "sparse6":
		return benchPairSeed(shape, n, seed)
	case "runs": // alternating disjoint runs; length and ownership rotate
		// per variant so the shape stays honest under branch prediction
		run := 8 + 8*r.Intn(4)
		swap := r.Intn(2) == 1
		a = make([]uint16, n)
		b = make([]uint16, n)
		for i := 0; i < n; i++ {
			blk, off := i/run, i%run
			av := uint16(blk*2*run + off)
			bv := uint16(blk*2*run + run + off)
			if swap {
				av, bv = bv, av
			}
			a[i], b[i] = av, bv
		}
		return a, b
	case "coinflip": // disjoint 8-blocks, ownership shuffled: zero matches,
		// skip direction unpredictable for the full length of both sides
		nb := (n + 7) / 8
		own := make([]bool, 2*nb)
		for i := 0; i < nb; i++ {
			own[i] = true
		}
		r.Shuffle(len(own), func(i, j int) { own[i], own[j] = own[j], own[i] })
		a = make([]uint16, 0, nb*8)
		b = make([]uint16, 0, nb*8)
		base := 0
		for _, toA := range own {
			for x := 0; x < 8; x++ {
				if toA {
					a = append(a, uint16(base+x))
				} else {
					b = append(b, uint16(base+x))
				}
			}
			base += 8
		}
		return a[:n], b[:n]
	case "skew8": // one side 8x longer, same value range: stays below the
		// 64:1 galloping cutoff, exercises the fast-forward paths
		big := 8 * n
		if big > 32768 {
			big = 32768
		}
		return genSortedUnique(r, n, 65536), genSortedUnique(r, big, 65536)
	case "overlap95": // ~95% shared elements: near-total match density
		a = genSortedUnique(r, n, 4*n)
		b = append([]uint16(nil), a...)
		for i := 10; i < n; i += 20 {
			b[i] ^= 1
		}
		seen := make(map[uint16]bool, n)
		out := b[:0]
		for _, v := range b {
			if !seen[v] {
				seen[v] = true
				out = append(out, v)
			}
		}
		b = out
		sortNeeded := false
		for i := 1; i < len(b); i++ {
			if b[i] < b[i-1] {
				sortNeeded = true
				break
			}
		}
		if sortNeeded {
			for i := 1; i < len(b); i++ {
				for j := i; j > 0 && b[j] < b[j-1]; j-- {
					b[j], b[j-1] = b[j-1], b[j]
				}
			}
		}
		return a, b
	}
	panic("unknown shape")
}

var benchIntersectShapes = []string{"dense50", "sparse6", "runs", "coinflip", "skew8", "overlap95"}
var benchIntersectSizes = []int{8, 16, 24, 32, 64, 128, 256, 512, 1001, 4096}

func BenchmarkIntersect2By2(b *testing.B) {
	for _, shape := range benchIntersectShapes {
		for _, n := range benchIntersectSizes {
			as := make([][]uint16, benchVariants)
			bs := make([][]uint16, benchVariants)
			for v := 0; v < benchVariants; v++ {
				as[v], bs[v] = benchIntersectPair(shape, n, int64(v))
			}
			// andArray allocates exactly min(len1,len2); mirror that cap.
			mins := make([]int, benchVariants)
			for v := 0; v < benchVariants; v++ {
				mins[v] = len(as[v])
				if len(bs[v]) < mins[v] {
					mins[v] = len(bs[v])
				}
			}
			buffer := make([]uint16, n+8)
			scratch := make([]uint16, n+8)
			for _, impl := range []struct {
				name string
				fn   func([]uint16, []uint16, []uint16) int
			}{
				{"dispatch", intersection2by2},
				{"scalar", localintersect2by2},
			} {
				b.Run(fmt.Sprintf("%s/%d/%s", shape, n, impl.name), func(b *testing.B) {
					sink := 0
					for i := 0; i < b.N; i++ {
						v := i % benchVariants
						sink += impl.fn(as[v], bs[v], buffer[:0:mins[v]])
					}
					_ = sink
				})
			}
			// In-place rows model iandArray: output aliases set1, so each
			// iteration pays one copy to restore the input; both rows pay
			// it, keeping the ratio honest.
			for _, impl := range []struct {
				name string
				fn   func([]uint16, []uint16, []uint16) int
			}{
				{"inplace", intersection2by2},
				{"inplaceScalar", localintersect2by2},
			} {
				b.Run(fmt.Sprintf("%s/%d/%s", shape, n, impl.name), func(b *testing.B) {
					sink := 0
					for i := 0; i < b.N; i++ {
						v := i % benchVariants
						m := copy(scratch, as[v])
						sink += impl.fn(scratch[:m], bs[v], scratch[:0:m])
					}
					_ = sink
				})
			}
		}
	}
}

func BenchmarkIntersectCard2By2(b *testing.B) {
	for _, shape := range benchIntersectShapes {
		for _, n := range benchIntersectSizes {
			as := make([][]uint16, benchVariants)
			bs := make([][]uint16, benchVariants)
			for v := 0; v < benchVariants; v++ {
				as[v], bs[v] = benchIntersectPair(shape, n, int64(v))
			}
			for _, impl := range []struct {
				name string
				fn   func([]uint16, []uint16) int
			}{
				{"dispatch", intersection2by2Cardinality},
				{"scalar", localintersect2by2Cardinality},
			} {
				b.Run(fmt.Sprintf("%s/%d/%s", shape, n, impl.name), func(b *testing.B) {
					sink := 0
					for i := 0; i < b.N; i++ {
						v := i % benchVariants
						sink += impl.fn(as[v], bs[v])
					}
					_ = sink
				})
			}
		}
	}
}
