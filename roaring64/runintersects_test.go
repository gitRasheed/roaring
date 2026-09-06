package roaring64

import "testing"

func TestIntersectsRunContainers(t *testing.T) {
	const base uint64 = 1 << 40
	x := New()
	x.AddRange(base+63, base+129)
	x.RunOptimize()
	for _, values := range [][]uint64{{}, {base + 62}, {base + 63}, {base + 128}, {base + 129}, {base + (1 << 32) + 63}} {
		y := BitmapOf(values...)
		want := !And(x, y).IsEmpty()
		beforeX, beforeY := x.Clone(), y.Clone()
		if x.Intersects(y) != want || y.Intersects(x) != want {
			t.Fatalf("values %v", values)
		}
		if n := testing.AllocsPerRun(10, func() { x.Intersects(y); y.Intersects(x) }); n != 0 {
			t.Fatalf("%g allocations", n)
		}
		if !x.Equals(beforeX) || !y.Equals(beforeY) {
			t.Fatal("modified input")
		}
	}
}
