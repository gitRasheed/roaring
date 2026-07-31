//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"fmt"
	"math/rand"
	"testing"
)

// Deterministic dataset shapes matching the C prototype's benchmark harness:
// dense50 (~50% overlap), sparse6 (~6%), runs16 (alternating 16-runs),
// dupheavy (b = a with every 8th element changed).
func benchPairSeed(shape string, n int, seed int64) (a, b []uint16) {
	r := rand.New(rand.NewSource(int64(42+n) + seed*7919))
	switch shape {
	case "dense50":
		return genSortedUnique(r, n, 2*n), genSortedUnique(r, n, 2*n)
	case "sparse6":
		vr := n * 16
		if vr > 65536 {
			vr = 65536
		}
		return genSortedUnique(r, n, vr), genSortedUnique(r, n, vr)
	case "runs16":
		a = make([]uint16, n)
		b = make([]uint16, n)
		for i := 0; i < n; i++ {
			blk, off := i/16, i%16
			a[i] = uint16(blk*32 + off)
			b[i] = uint16(blk*32 + 16 + off)
		}
		return a, b
	case "dupheavy":
		a = genSortedUnique(r, n, 2*n)
		b = make([]uint16, n)
		copy(b, a)
		// replace every 8th with a value above a's range so b stays
		// duplicate-free after re-sorting (~87.5% overlap with a)
		for i := 0; i < n; i += 8 {
			b[i] = uint16(2*n + i)
		}
		for i := 1; i < n; i++ {
			for j := i; j > 0 && b[j] < b[j-1]; j-- {
				b[j], b[j-1] = b[j-1], b[j]
			}
		}
		return a, b
	}
	panic("unknown shape")
}

// Each config cycles through 16 dataset variants (all fixed-seed, so runs are
// deterministic): benchmarking a single fixed pair lets the branch predictor
// memorize the scalar path's decision sequence and report unrealistically
// fast scalar times on wide cores.
const benchVariants = 16

func BenchmarkUnion2By2(bench *testing.B) {
	for _, shape := range []string{"dense50", "sparse6", "runs16", "dupheavy"} {
		for _, n := range []int{64, 256, 1024, 4096} {
			as := make([][]uint16, benchVariants)
			bs := make([][]uint16, benchVariants)
			for v := 0; v < benchVariants; v++ {
				as[v], bs[v] = benchPairSeed(shape, n, int64(v))
			}
			buffer := make([]uint16, 2*n)
			for _, impl := range []struct {
				name string
				fn   func([]uint16, []uint16, []uint16) int
			}{
				{"neon", union2by2},
				{"scalar", union2by2scalar},
			} {
				bench.Run(fmt.Sprintf("%s/%d/%s", shape, n, impl.name), func(bench *testing.B) {
					sink := 0
					for i := 0; i < bench.N; i++ {
						v := i % benchVariants
						sink += impl.fn(as[v], bs[v], buffer)
					}
					_ = sink
				})
			}
		}
	}
}
