//go:build arm64 && !gccgo && !appengine
// +build arm64,!gccgo,!appengine

package roaring

// Study-only bench: forces the NEON intersect path below the shipping
// threshold to locate the true crossover. Not for shipping.

import (
	"fmt"
	"testing"
)

func intersectNEONStudy(set1, set2, buffer []uint16) int {
	if set1[len(set1)-1] < set2[0] || set2[len(set2)-1] < set1[0] {
		return 0
	}
	buffer = buffer[:cap(buffer)]
	var spill [8]uint16
	outLen, pos1, pos2, spilled := intersectKernelNEON(set1, set2, buffer, &uniqshuf[0], &spill)
	if spilled != 0 {
		i := 0
		for i < 8 && pos2 < len(set2) {
			switch {
			case spill[i] < set2[pos2]:
				i++
			case set2[pos2] < spill[i]:
				pos2++
			default:
				buffer[outLen] = spill[i]
				outLen++
				i++
				pos2++
			}
		}
	}
	return outLen + localintersect2by2(set1[pos1:], set2[pos2:], buffer[outLen:])
}

func intersectCardNEONStudy(set1, set2 []uint16) int {
	if set1[len(set1)-1] < set2[0] || set2[len(set2)-1] < set1[0] {
		return 0
	}
	card, pos1, pos2 := intersectCardKernelNEON(set1, set2)
	return card + localintersect2by2Cardinality(set1[pos1:], set2[pos2:])
}

var studyShapes = []string{"dense50", "sparse6", "runs", "coinflip", "skew8", "overlap95"}
var studySizes = []int{8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64}

func TestIntersectStudyMatchesScalar(t *testing.T) {
	for _, shape := range studyShapes {
		for _, n := range studySizes {
			for seed := int64(0); seed < 16; seed++ {
				a, b := benchIntersectPair(shape, n, seed)
				if len(a) < 8 || len(b) < 8 {
					continue
				}
				m := len(a)
				if len(b) < m {
					m = len(b)
				}
				want := make([]uint16, m)
				wn := localintersect2by2(a, b, want)
				got := make([]uint16, m)
				gn := intersectNEONStudy(a, b, got[:0:m])
				if gn != wn {
					t.Fatalf("%s/%d/%d: len want %d got %d", shape, n, seed, wn, gn)
				}
				for i := 0; i < wn; i++ {
					if got[i] != want[i] {
						t.Fatalf("%s/%d/%d: idx %d want %d got %d", shape, n, seed, i, want[i], got[i])
					}
				}
				if c := intersectCardNEONStudy(a, b); c != wn {
					t.Fatalf("%s/%d/%d: card want %d got %d", shape, n, seed, wn, c)
				}
			}
		}
	}
}

func BenchmarkIntersectSmallN(b *testing.B) {
	const variants = 16
	for _, shape := range studyShapes {
		for _, n := range studySizes {
			as := make([][]uint16, variants)
			bs := make([][]uint16, variants)
			bufs := make([][]uint16, variants)
			for v := 0; v < variants; v++ {
				as[v], bs[v] = benchIntersectPair(shape, n, int64(v))
				m := len(as[v])
				if len(bs[v]) < m {
					m = len(bs[v])
				}
				bufs[v] = make([]uint16, m)
			}
			run := func(name string, fn func(v int) int) {
				b.Run(fmt.Sprintf("%s/%d/%s", shape, n, name), func(b *testing.B) {
					sink := 0
					for i := 0; i < b.N; i++ {
						sink += fn(i & (variants - 1))
					}
					sinkHole = sink
				})
			}
			run("kernel", func(v int) int {
				return intersectNEONStudy(as[v], bs[v], bufs[v][:0:cap(bufs[v])])
			})
			run("scalar", func(v int) int {
				return localintersect2by2(as[v], bs[v], bufs[v])
			})
			run("kernelcard", func(v int) int {
				return intersectCardNEONStudy(as[v], bs[v])
			})
			run("scalarcard", func(v int) int {
				return localintersect2by2Cardinality(as[v], bs[v])
			})
		}
	}
}

var sinkHole int
