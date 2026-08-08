//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"fmt"
	"math/rand"
	"testing"
)

// Study-only bench: kernel (unionNEON, threshold-independent) vs scalar asm
// across small sizes never measured on the shipped kernel. Not for shipping.
func studyPairSeed(shape string, n int, seed int64) (a, b []uint16) {
	r := rand.New(rand.NewSource(int64(42+n) + seed*7919))
	switch shape {
	case "dense50", "sparse6", "runs16", "spread":
		return benchPairSeed(shape, n, seed)
	case "identical":
		a = genSortedUnique(r, n, 2*n)
		b = make([]uint16, n)
		copy(b, a)
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

func TestStudyKernelMatchesScalar(t *testing.T) {
	shapes := []string{"dense50", "sparse6", "runs16", "spread", "identical", "dupheavy"}
	sizes := []int{8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512}
	for _, shape := range shapes {
		for _, n := range sizes {
			for v := int64(0); v < benchVariants; v++ {
				a, b := studyPairSeed(shape, n, v)
				bufK := make([]uint16, len(a)+len(b))
				bufS := make([]uint16, len(a)+len(b))
				gotK := bufK[:unionNEON(a, b, bufK)]
				gotS := bufS[:union2by2scalar(a, b, bufS)]
				if len(gotK) != len(gotS) {
					t.Fatalf("%s/%d/v%d: len %d vs %d", shape, n, v, len(gotK), len(gotS))
				}
				for i := range gotK {
					if gotK[i] != gotS[i] {
						t.Fatalf("%s/%d/v%d: idx %d: %d vs %d", shape, n, v, i, gotK[i], gotS[i])
					}
				}
			}
		}
	}
}

func BenchmarkUnionSmallN(b *testing.B) {
	for _, shape := range []string{"dense50", "sparse6", "runs16", "spread", "identical", "dupheavy"} {
		for _, n := range []int{8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512} {
			as := make([][]uint16, benchVariants)
			bs := make([][]uint16, benchVariants)
			for v := 0; v < benchVariants; v++ {
				as[v], bs[v] = studyPairSeed(shape, n, int64(v))
			}
			buffer := make([]uint16, 2*n)
			for _, impl := range []struct {
				name string
				fn   func([]uint16, []uint16, []uint16) int
			}{
				{"kernel", unionNEON},
				{"scalar", union2by2scalar},
			} {
				b.Run(fmt.Sprintf("%s/%d/%s", shape, n, impl.name), func(b *testing.B) {
					sink := 0
					for i := 0; i < b.N; i++ {
						v := i % benchVariants
						sink += impl.fn(as[v], bs[v], buffer)
					}
					_ = sink
				})
			}
		}
	}
}
