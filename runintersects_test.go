package roaring

import (
	"math/rand"
	"testing"
)

func TestRunIntersects(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for trial := 0; trial < 200; trial++ {
		runs := newRunContainer16()
		array := newArrayContainer()
		bitmap := newBitmapContainer()
		other := newRunContainer16()
		for i := 0; i < 64; i++ {
			start := r.Intn(65536)
			runs.iaddRange(start, min(65536, start+r.Intn(512)+1))
			start = r.Intn(65536)
			other.iaddRange(start, min(65536, start+r.Intn(512)+1))
			array.iadd(uint16(r.Intn(65536)))
			bitmap.iadd(uint16(r.Intn(65536)))
		}
		for _, c := range []container{array, bitmap, other, runs} {
			want := !runs.and(c).isEmpty()
			before, beforeOther := runs.clone(), c.clone()
			if runs.intersects(c) != want || c.intersects(runs) != want {
				t.Fatalf("trial %d, %T: incorrect overlap", trial, c)
			}
			if n := testing.AllocsPerRun(10, func() { runs.intersects(c); c.intersects(runs) }); n != 0 {
				t.Fatalf("%T: %g allocations", c, n)
			}
			if !runs.equals(before) || !c.equals(beforeOther) {
				t.Fatal("modified input")
			}
		}
	}
}

func TestRunIntersectsBoundaries(t *testing.T) {
	for _, start := range []int{0, 1, 63, 64, 65, 127, 128, 65534, 65535} {
		for _, end := range []int{start + 1, min(start+64, 65536), 65536} {
			runs := newRunContainer16()
			runs.iaddRange(start, end)
			for _, value := range []int{start - 1, start, end - 1, end} {
				if value < 0 || value >= 65536 {
					continue
				}
				array := newArrayContainer()
				array.iadd(uint16(value))
				bitmap := newBitmapContainer()
				bitmap.iadd(uint16(value))
				other := newRunContainer16()
				other.iaddRange(value, value+1)
				for _, c := range []container{array, bitmap, other} {
					want := value >= start && value < end
					if runs.intersects(c) != want || c.intersects(runs) != want {
						t.Fatalf("[%d,%d), value %d, %T", start, end, value, c)
					}
				}
			}
		}
	}
	// Interleaved containers have overlapping bounds but no common values.
	x, y := newRunContainer16(), newRunContainer16()
	array, bitmap := newArrayContainer(), newBitmapContainer()
	for v := 0; v < 65536; v += 64 {
		x.iv = append(x.iv, newInterval16Range(uint16(v), uint16(v+15)))
		y.iv = append(y.iv, newInterval16Range(uint16(v+32), uint16(v+47)))
		array.iadd(uint16(v + 32))
		bitmap.iadd(uint16(v + 32))
	}
	for _, c := range []container{y, array, bitmap} {
		if x.intersects(c) || c.intersects(x) {
			t.Fatal("interleaved containers intersect")
		}
		c.iadd(65535)
		x.iadd(65535)
		if !x.intersects(c) || !c.intersects(x) {
			t.Fatal("missed last-value overlap")
		}
		x.iremove(65535)
	}
	empty := newRunContainer16()
	full := newRunContainer16()
	full.iaddRange(0, 65536)
	for _, c := range []container{newArrayContainer(), newBitmapContainer(), empty, full} {
		if empty.intersects(c) || c.intersects(empty) {
			t.Fatal("empty container intersects")
		}
	}
}
