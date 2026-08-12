//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

import (
	"fmt"
	"math/rand"
	"testing"
)

// Shapes that separate the partition kernel from the carry kernel: the
// overlap fraction decides how many elements one iteration consumes.
func partPairSeed(shape string, n int, seed int64) (a, b []uint16) {
	r := rand.New(rand.NewSource(int64(1013+n) + seed*104729))
	switch shape {
	case "interleaved": // strict alternation, no shared values
		base := r.Intn(65536 - 2*n - 2)
		a = make([]uint16, n)
		b = make([]uint16, n)
		for i := 0; i < n; i++ {
			a[i] = uint16(base + 2*i)
			b[i] = uint16(base + 2*i + 1)
		}
		return a, b
	case "shared7of8":
		return sharedPair(r, n, 7, 8)
	case "shared95":
		return sharedPair(r, n, 19, 20)
	case "runs16": // 16-element runs, alternating ownership
		a = make([]uint16, n)
		b = make([]uint16, n)
		for i := 0; i < n; i++ {
			blk, off := i/16, i%16
			a[i] = uint16(blk*32 + off)
			b[i] = uint16(blk*32 + 16 + off)
		}
		return a, b
	case "random":
		return genSortedUnique(r, n, 65536), genSortedUnique(r, n, 65536)
	case "disjointcoin": // ownership of each run drawn per run, not alternating
		a = make([]uint16, 0, n)
		b = make([]uint16, 0, n)
		v := 0
		for len(a) < n || len(b) < n {
			own := r.Intn(2) == 0
			run := 48 + r.Intn(33)
			for i := 0; i < run && v < 65535; i++ {
				if own && len(a) < n {
					a = append(a, uint16(v))
				} else if !own && len(b) < n {
					b = append(b, uint16(v))
				} else if len(a) < n {
					a = append(a, uint16(v))
				} else if len(b) < n {
					b = append(b, uint16(v))
				} else {
					break
				}
				v++
			}
		}
		return a, b
	case "runs4", "runs8", "runs32":
		rl := map[string]int{"runs4": 4, "runs8": 8, "runs32": 32}[shape]
		a = make([]uint16, 0, n)
		b = make([]uint16, 0, n)
		v := 0
		for len(a) < n || len(b) < n {
			for i := 0; i < rl && len(a) < n; i++ {
				a = append(a, uint16(v))
				v++
			}
			for i := 0; i < rl && len(b) < n; i++ {
				b = append(b, uint16(v))
				v++
			}
		}
		return a, b
	case "tileshuf", "tileper":
		// 384 tiles of 16 values, a third owned by each side and a third
		// shared. Shuffled ownership gives both window branches entropy; the
		// periodic control has identical path counts and identical overlap.
		tiles := 3 * n / 32
		tag := make([]byte, tiles)
		for i := 0; i < tiles; i++ {
			tag[i] = byte(i % 3)
		}
		if shape == "tileshuf" {
			r.Shuffle(tiles-1, func(i, j int) { tag[i], tag[j] = tag[j], tag[i] })
		}
		for t := 0; t < tiles; t++ {
			for i := 0; i < 16; i++ {
				v := uint16(16*t + i)
				if tag[t] != 1 {
					a = append(a, v)
				}
				if tag[t] != 0 {
					b = append(b, v)
				}
			}
		}
		return a, b
	case "minprogress":
		// Every window pair defeats both shortcuts by equality, and the
		// general path advances the minimum 17 elements.
		for k := 0; len(a) < n-16 && len(b) < n-16; k++ {
			base := 32 * k
			for i := 0; i <= 15; i++ {
				a = append(a, uint16(base+i))
			}
			a = append(a, uint16(base+31))
			for i := 15; i <= 31; i++ {
				b = append(b, uint16(base+i))
			}
		}
		last := int(a[len(a)-1]) + 1
		for i := 0; len(a) < n; i++ {
			a = append(a, uint16(last+i))
		}
		last = int(a[len(a)-1])
		for i := 0; len(b) < n; i++ {
			b = append(b, uint16(last+i))
		}
		return a, b
	case "identical":
		a = make([]uint16, n)
		for i := range a {
			a[i] = uint16(i)
		}
		return a, a
	}
	// dense50, sparse6, spread come from the shipped benchmark's generator.
	return benchPairSeed(shape, n, seed)
}

// num/den of each array is shared with the other; ownership is shuffled so
// the scalar merge cannot predict the decision sequence from position.
func sharedPair(r *rand.Rand, n, num, den int) (a, b []uint16) {
	nc := n * num / den
	nu := n - nc
	pool := genSortedUnique(r, nc+2*nu, 65536)
	tag := make([]byte, len(pool))
	for i := 0; i < len(pool); i++ {
		switch {
		case i < nc:
			tag[i] = 2
		case i < nc+nu:
			tag[i] = 0
		default:
			tag[i] = 1
		}
	}
	r.Shuffle(len(tag), func(i, j int) { tag[i], tag[j] = tag[j], tag[i] })
	for i, v := range pool {
		if tag[i] != 1 {
			a = append(a, v)
		}
		if tag[i] != 0 {
			b = append(b, v)
		}
	}
	return a, b
}

func BenchmarkUnionPartition(b *testing.B) {
	shapes := []string{"interleaved", "shared7of8", "shared95", "random", "disjointcoin", "runs16", "dense50", "spread",
		"runs4", "runs8", "runs32", "tileshuf", "tileper", "minprogress", "identical"}
	for _, shape := range shapes {
		for _, n := range []int{512, 4096} {
			as := make([][]uint16, benchVariants)
			bs := make([][]uint16, benchVariants)
			for v := 0; v < benchVariants; v++ {
				as[v], bs[v] = partPairSeed(shape, n, int64(v))
			}
			buffer := make([]uint16, 2*n+64)
			for _, impl := range []struct {
				name string
				fn   func([]uint16, []uint16, []uint16) int
			}{
				{"part", unionNEONPart},
				{"carry", unionNEON},
				{"scalar", union2by2scalar},
			} {
				b.Run(fmt.Sprintf("%s/%d/%s", shape, n, impl.name), func(b *testing.B) {
					sink := 0
					elems := 0
					for v := 0; v < benchVariants; v++ {
						elems += len(as[v]) + len(bs[v])
					}
					perCall := float64(elems) / float64(benchVariants)
					for i := 0; i < b.N; i++ {
						v := i % benchVariants
						sink += impl.fn(as[v], bs[v], buffer[:0:len(buffer)])
					}
					b.ReportMetric(float64(b.Elapsed().Nanoseconds())/(float64(b.N)*perCall), "ns/elem")
					_ = sink
				})
			}
		}
	}
}
