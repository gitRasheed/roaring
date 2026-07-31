package roaring

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/mschoch/smat"
)

// Native-fuzzing entry for the smat state machine, matching the workflow
// documented in smat.go. Seeds from the workdir corpus when present.
func FuzzSmat(f *testing.F) {
	if entries, err := os.ReadDir("workdir/corpus"); err == nil {
		for _, e := range entries {
			if data, err := os.ReadFile(filepath.Join("workdir/corpus", e.Name())); err == nil {
				f.Add(data)
			}
		}
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		smat.Fuzz(&smatContext{}, smat.ActionID('S'), smat.ActionID('T'), smatActionMap, data)
	})
}
