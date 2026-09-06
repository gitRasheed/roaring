package roaring

import (
	"math/rand"
	"testing"
)

func BenchmarkRunIntersects(b *testing.B) {
	for _, shape := range []string{"array", "bitmap", "run"} {
		for _, match := range []string{"early", "late", "none"} {
			b.Run(shape+"/"+match, func(b *testing.B) {
				var left, right [64]*Bitmap
				var want [64]bool
				rng := rand.New(rand.NewSource(42))
				for i := range left {
					x := newRunContainer16()
					for start := 0; start < 32768; start += 512 {
						lo := start + rng.Intn(128)
						x.iv = append(x.iv, newInterval16Range(uint16(lo), uint16(lo+127)))
					}
					y := New()
					switch shape {
					case "array":
						for j := 0; j < 512; j++ {
							y.Add(uint32(32768 + rng.Intn(32768)))
						}
					case "bitmap":
						for j := 32768; j < 65536; j += 2 {
							y.Add(uint32(j))
						}
					case "run":
						for j := 33000; j < 65000; j += 512 {
							y.AddRange(uint64(j), uint64(j+128))
						}
					}
					if match == "early" {
						y.Add(uint32(x.iv[0].start))
					}
					if match == "late" {
						y.Add(uint32(x.iv[len(x.iv)-1].last()))
					}
					left[i] = New()
					left[i].highlowcontainer.appendContainer(0, x, false)
					right[i] = y
					want[i] = !And(left[i], right[i]).IsEmpty()
				}
				var got bool
				i := 0
				b.ReportAllocs()
				for b.Loop() {
					got = left[i].Intersects(right[i])
					i = (i + 1) % len(left)
				}
				if got != want[(i+len(left)-1)%len(left)] {
					b.Fatal("incorrect overlap")
				}
			})
		}
	}
}

func BenchmarkRunIntersectsSkew(b *testing.B) {
	x := newRunContainer16()
	for v := 0; v < 65500; v += 32 {
		x.iv = append(x.iv, newInterval16Range(uint16(v), uint16(v+7)))
	}
	var others [64]*runContainer16
	rng := rand.New(rand.NewSource(77))
	for i := range others {
		v := uint16(32*(512+rng.Intn(1024)) + 16)
		others[i] = newRunContainer16Range(v, v+3)
	}
	var got bool
	i := 0
	b.ReportAllocs()
	for b.Loop() {
		got = x.intersects(others[i])
		i = (i + 1) % len(others)
	}
	if got {
		b.Fatal("disjoint runs intersect")
	}
}
