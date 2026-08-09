//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

// Study-only bench: forces the NEON andnot path below the shipping
// threshold to locate the true crossover. Not for shipping.

import (
	"fmt"
	"math/rand"
	"testing"
)

func andnotNEONStudy(set1, set2, buffer []uint16) int {
	if set1[len(set1)-1] < set2[0] || set2[len(set2)-1] < set1[0] {
		buffer = buffer[:len(set1)]
		copy(buffer, set1)
		return len(set1)
	}
	buffer = buffer[:cap(buffer)]
	var spill [8]uint16
	outLen, pos1, pos2, spilled, accMask := andnotKernelNEON(set1, set2, buffer, &uniqshuf[0], &spill)
	if spilled != 0 {
		i := 0
		for i < 8 && pos2 < len(set2) {
			switch {
			case spill[i] < set2[pos2]:
				if accMask&(1<<i) == 0 {
					buffer[outLen] = spill[i]
					outLen++
				}
				i++
			case set2[pos2] < spill[i]:
				pos2++
			default:
				i++
				pos2++
			}
		}
		for ; i < 8; i++ {
			if accMask&(1<<i) == 0 {
				buffer[outLen] = spill[i]
				outLen++
			}
		}
	}
	return outLen + localdifference(set1[pos1:], set2[pos2:], buffer[outLen:])
}

var andnotStudyShapes = []string{"dense50", "sparse6", "runs", "coinflip", "skew8", "pickoff", "ident"}
var andnotStudySizes = []int{8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96}

func andnotStudyPair(shape string, n int, seed int64) ([]uint16, []uint16) {
	r := rand.New(rand.NewSource(seed*7919 + int64(n)))
	switch shape {
	case "dense50":
		return genSortedUnique(r, n, 2*n), genSortedUnique(r, n, 2*n)
	case "sparse6":
		return genSortedUnique(r, n, 16*n), genSortedUnique(r, n, 16*n)
	case "runs":
		a := make([]uint16, n)
		b := make([]uint16, n)
		for i := 0; i < n; i++ {
			blk, off := i/16, i%16
			a[i] = uint16(blk*32 + off)
			b[i] = uint16(blk*32 + 16 + off)
		}
		return a, b
	case "coinflip":
		a := make([]uint16, 0, n)
		b := make([]uint16, 0, n)
		base := 0
		for len(a) < n || len(b) < n {
			toA := len(a) < n && (len(b) >= n || r.Intn(2) == 0)
			for x := 0; x < 8; x++ {
				if toA && len(a) < n {
					a = append(a, uint16(base+x))
				}
				if !toA && len(b) < n {
					b = append(b, uint16(base+x))
				}
			}
			base += 8
		}
		return a, b
	case "skew8":
		return genSortedUnique(r, n, 60000), genSortedUnique(r, 8*n, 60000)
	case "pickoff":
		// deletion masking: b removes every third element of a
		a := make([]uint16, n)
		for i := range a {
			a[i] = uint16(i * 3)
		}
		var b []uint16
		for i := 1; i < n; i += 3 {
			b = append(b, uint16(i*3))
		}
		return a, b
	case "ident":
		// annihilation: AndNot against an equal set empties the result
		a := genSortedUnique(r, n, 4*n)
		return a, append([]uint16(nil), a...)
	}
	panic(shape)
}

func TestAndnotStudyMatchesScalar(t *testing.T) {
	for _, shape := range andnotStudyShapes {
		for _, n := range andnotStudySizes {
			for seed := int64(0); seed < 16; seed++ {
				a, b := andnotStudyPair(shape, n, seed)
				for o := 0; o < 2; o++ {
					if o == 1 {
						a, b = b, a
					}
					if len(a) < 8 || len(b) < 8 {
						continue
					}
					want := make([]uint16, len(a))
					wn := localdifference(a, b, want)
					got := make([]uint16, len(a))
					gn := andnotNEONStudy(a, b, got[:0:len(a)])
					if gn != wn {
						t.Fatalf("%s/%d/%d/o%d: len want %d got %d", shape, n, seed, o, wn, gn)
					}
					for i := 0; i < wn; i++ {
						if got[i] != want[i] {
							t.Fatalf("%s/%d/%d/o%d: idx %d want %d got %d", shape, n, seed, o, i, want[i], got[i])
						}
					}
				}
			}
		}
	}
}

func BenchmarkAndnotSmallN(b *testing.B) {
	const variants = 16
	for _, shape := range andnotStudyShapes {
		for _, n := range andnotStudySizes {
			as := make([][]uint16, variants)
			bs := make([][]uint16, variants)
			bufs := make([][]uint16, variants)
			ok := true
			for v := 0; v < variants; v++ {
				as[v], bs[v] = andnotStudyPair(shape, n, int64(v))
				if len(as[v]) < 8 || len(bs[v]) < 8 {
					ok = false
					break
				}
				bufs[v] = make([]uint16, len(as[v]))
			}
			if !ok {
				continue
			}
			run := func(name string, fn func(v int) int) {
				b.Run(fmt.Sprintf("%s/%d/%s", shape, n, name), func(b *testing.B) {
					sink := 0
					for i := 0; i < b.N; i++ {
						sink += fn(i & (variants - 1))
					}
					andnotSinkHole = sink
				})
			}
			run("kernel", func(v int) int {
				return andnotNEONStudy(as[v], bs[v], bufs[v][:0:cap(bufs[v])])
			})
			run("scalar", func(v int) int {
				return localdifference(as[v], bs[v], bufs[v])
			})
		}
	}
}

var andnotSinkHole int
