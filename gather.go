package astiocr

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/asticode/go-astilog"
	"github.com/golang/freetype/truetype"
	"github.com/pkg/errors"
	ft "golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// GatherSummary represents a gather summary
type GatherSummary struct {
	Images []GatherSummaryImage `json:"images"`
}

// GatherSummaryImage represents a gather summary image
type GatherSummaryImage struct {
	Height int                `json:"height"`
	Boxes  []GatherSummaryBox `json:"boxes"`
	Path   string             `json:"path"`
	Width  int                `json:"width"`
}

// GatherSummaryBox represents a gather summary box
type GatherSummaryBox struct {
	Label      string `json:"label"`
	LabelIndex int    `json:"label_index"`
	X0         int    `json:"x0"`
	X1         int    `json:"x1"`
	Y0         int    `json:"y0"`
	Y1         int    `json:"y1"`
}

// Gather gathers training data
func (t *Trainer) Gather(ctx context.Context) (err error) {
	// Init
	rand.Seed(time.Now().UnixNano())

	// Create data folders
	if err = t.createDataFolders(); err != nil {
		err = errors.Wrap(err, "astiocr: creating data folders failed")
		return
	}

	// Create label map
	if err = t.createLabelMap(); err != nil {
		err = errors.Wrap(err, "astiocr: creating label map failed")
		return
	}

	// Loop through count
	var summaryTraining, summaryTest GatherSummary
	astilog.Debugf("astiocr: generating %d images (%d for training - %d for test)", t.count, t.trainingDataCount, t.testDataCount)
	for idx := 0; idx < t.count; idx++ {
		// Check context
		if ctx.Err() != nil {
			err = errors.Wrap(err, "astiocr: context error")
			return
		}

		// Create image
		var img *image.RGBA
		var si GatherSummaryImage
		if idx < t.trainingDataCount {
			img, si = t.createImageStrategy2()
		} else {
			img, si = t.createImageStrategy2()
		}

		// No boxes
		if len(si.Boxes) == 0 {
			continue
		}

		// Store image
		var p string
		if p, err = t.storeImage(idx, img); err != nil {
			err = errors.Wrap(err, "astiocr: storing image failed")
			return
		}
		si.Path = p

		// Append image to summary
		if idx < t.trainingDataCount {
			summaryTraining.Images = append(summaryTraining.Images, si)
		} else {
			summaryTest.Images = append(summaryTest.Images, si)
		}

		// Log
		if (idx+1)%50 == 0 && idx > 0 {
			astilog.Debugf("astiocr: %d/%d images created", idx+1, t.count)
		}
	}

	// Write summaries
	if err = t.writeSummaries(summaryTraining, summaryTest); err != nil {
		err = errors.Wrap(err, "astiocr: writing summary failed")
		return
	}

	// Prepare data
	if err = t.prepareData(ctx); err != nil {
		err = errors.Wrap(err, "astiocr: preparing data failed")
		return
	}
	return
}

func (t *Trainer) createDataFolders() (err error) {
	// Remove folder
	astilog.Debugf("astiocr: removing %s", t.outputDataDirectoryPath)
	if err = os.RemoveAll(t.outputDataDirectoryPath); err != nil {
		err = errors.Wrapf(err, "astiocr: removeAll %s failed", t.outputDataDirectoryPath)
		return
	}

	// Loop through folders to create
	for _, p := range []string{
		filepath.Join(t.outputDataDirectoryPath, "images"),
		filepath.Join(t.outputDataDirectoryPath, "test"),
		filepath.Join(t.outputDataDirectoryPath, "training"),
	} {
		astilog.Debugf("astiocr: creating %s", p)
		if err = os.MkdirAll(p, 0700); err != nil {
			err = errors.Wrapf(err, "astiocr: mkdirall %s failed", p)
		}
	}
	return
}

const characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func (t *Trainer) createLabelMap() (err error) {
	// Create file
	var f *os.File
	p := filepath.Join(t.outputDataDirectoryPath, "label_map.pbtxt")
	if f, err = os.Create(p); err != nil {
		err = errors.Wrapf(err, "astiocr: creating %s failed", p)
		return
	}
	defer f.Close()

	// Loop through characters
	astilog.Debugf("astiocr: creating label map to %s", p)
	for idx, c := range characters {
		if c == 'E' || c == 'e' {
			if _, err = f.WriteString(fmt.Sprintf("item {\n  id: %d\n  name: '%s'\n}\n", idx+1, string(c))); err != nil {
				err = errors.Wrapf(err, "astiocr: writing to %s failed", p)
				return
			}
		}
	}
	return
}

func (t *Trainer) createImageStrategy1() (img *image.RGBA, si GatherSummaryImage) {
	// Initialize parameters
	fontSize, backgroundColor, fontColor, font := t.initParams()

	// Get coordinates
	x0, x1, y0, y1 := 0, int(float64(fontSize)*1.5), 0, int(float64(fontSize)*1.5)

	// Create image
	img, si = t.createImage(backgroundColor, y1, x1)

	// Draw character
	char, charIdx := t.drawCharacter(img, fontColor, font, fontSize, int(float64(fontSize)*0.3), int(float64(fontSize)*1.3))

	// Add box to summary
	si.Boxes = append(si.Boxes, GatherSummaryBox{
		Label:      string(char),
		LabelIndex: charIdx + 1,
		X0:         x0,
		X1:         x1,
		Y0:         y0,
		Y1:         y1,
	})
	return
}

func (t *Trainer) createImageStrategy2() (img *image.RGBA, si GatherSummaryImage) {
	// Initialize parameters
	fontSize, backgroundColor, fontColor, font := t.initParams()
	coverage := rand.Intn(50)

	// Create image
	img, si = t.createImage(backgroundColor, t.image.Height, t.image.Width)

	// Draw characters
	t.drawCharacters(fontSize, coverage, img, fontColor, &si, font)
	return
}

func (t *Trainer) initParams() (fontSize int, backgroundColor, fontColor color.RGBA, font *font) {
	fontSize = rand.Intn(6) + 12
	cc := t.colors[0]
	if len(t.colors) > 1 {
		cc = t.colors[rand.Intn(len(t.colors)-1)]
	}
	backgroundColor = cc.Background.RGBA
	fontColor = cc.Fonts[0].RGBA
	if len(cc.Fonts) > 1 {
		fontColor = cc.Fonts[rand.Intn(len(cc.Fonts)-1)].RGBA
	}
	font = t.fonts[0]
	if len(t.fonts) > 1 {
		font = t.fonts[rand.Intn(len(t.fonts)-1)]
	}
	return
}

func (t *Trainer) createImage(backgroundColor color.Color, height, width int) (img *image.RGBA, si GatherSummaryImage) {
	// Create image
	img = image.NewRGBA(image.Rect(0, 0, width, height))
	si = GatherSummaryImage{
		Height: height,
		Width:  width,
	}

	// Draw background
	draw.Draw(img, img.Bounds(), &image.Uniform{backgroundColor}, image.ZP, draw.Src)
	return
}

func (t *Trainer) drawCharacters(fontSize, coverage int, img *image.RGBA, fontColor color.RGBA, si *GatherSummaryImage, font *font) {
	step := fontSize
	for row := step; row < t.image.Height; row += step {
		// Loop through columns
		for col := 0; col+step < t.image.Width; col += step {
			// Get grid coordinates
			x0, x1, y0, y1 := col, col+step, row-step, row

			// Show grid
			if t.showGrid {
				t.drawBox(x0, x1, y0, y1, img, fontColor)
			}

			// Check coverage
			if rand.Intn(100) > coverage {
				continue
			}

			// Draw character
			char, charIdx := t.drawCharacter(img, fontColor, font, fontSize, col, row)

			// Only parse "e" letters for now
			if char == "E" || char == "e" {
				// Show box
				if t.showBox && !t.showGrid {
					t.drawBox(x0, x1, y0, y1, img, fontColor)
				}

				// Add box to summary
				si.Boxes = append(si.Boxes, GatherSummaryBox{
					Label:      string(char),
					LabelIndex: charIdx + 1,
					X0:         x0,
					X1:         x1,
					Y0:         y0,
					Y1:         y1,
				})
			}
		}
	}
	return
}

func (t *Trainer) drawBox(x0, x1, y0, y1 int, img draw.Image, c color.Color) {
	borderTop := image.Rect(x0, y0, x1, y0+1)
	borderRight := image.Rect(x1, y0, x1+1, y1)
	borderBottom := image.Rect(x0, y1, x1, y1+1)
	borderLeft := image.Rect(x0, y0, x0+1, y1)
	draw.Draw(img, borderTop, &image.Uniform{c}, image.ZP, draw.Src)
	draw.Draw(img, borderRight, &image.Uniform{c}, image.ZP, draw.Src)
	draw.Draw(img, borderBottom, &image.Uniform{c}, image.ZP, draw.Src)
	draw.Draw(img, borderLeft, &image.Uniform{c}, image.ZP, draw.Src)
}

func (t *Trainer) drawCharacter(img draw.Image, fontColor color.Color, font *font, fontSize, col, row int) (char string, charIdx int) {
	// Get character
	charIdx = rand.Intn(len(characters) - 1)
	char = string(characters[charIdx])

	// Draw character
	d := &ft.Drawer{
		Dst: img,
		Src: image.NewUniform(fontColor),
		Face: truetype.NewFace(font.font, &truetype.Options{
			DPI:  72,
			Size: float64(fontSize),
		}),
		Dot: fixed.P(col+int(float64(fontSize)/2.0/font.positionRatio), row-int(float64(fontSize)/2.0/font.positionRatio)),
	}
	d.DrawString(char)
	return
}

func (t *Trainer) storeImage(idx int, img *image.RGBA) (p string, err error) {
	// Create file
	var f *os.File
	p = filepath.Join(t.outputDataDirectoryPath, "images", fmt.Sprintf("%d.png", idx+1))
	if f, err = os.Create(p); err != nil {
		err = errors.Wrapf(err, "astiocr: creating %s failed", p)
		return
	}
	defer f.Close()

	// Encode image
	if err = png.Encode(f, img); err != nil {
		err = errors.Wrap(err, "astiocr: encoding image failed")
		return
	}
	return
}

func (t *Trainer) writeSummaries(summaryTraining, summaryTest GatherSummary) (err error) {
	for p, s := range map[string]GatherSummary{
		filepath.Join(t.outputDataDirectoryPath, "test", "summary.json"):     summaryTest,
		filepath.Join(t.outputDataDirectoryPath, "training", "summary.json"): summaryTraining,
	} {
		if err = t.writeSummary(s, p); err != nil {
			err = errors.Wrapf(err, "astiocr: writing summary to %s failed", p)
			return
		}
	}
	return
}

func (t *Trainer) writeSummary(s GatherSummary, p string) (err error) {
	// Create file
	var f *os.File
	if f, err = os.Create(p); err != nil {
		err = errors.Wrapf(err, "astiocr: creating %s failed", p)
		return
	}
	defer f.Close()

	// Write summary
	astilog.Debugf("astiocr: writing summary to %s", p)
	if err = json.NewEncoder(f).Encode(s); err != nil {
		err = errors.Wrap(err, "astiocr: writing summary failed")
		return
	}
	return
}

func (t *Trainer) prepareData(ctx context.Context) (err error) {
	cmd := exec.CommandContext(ctx, t.pythonBinaryPath, filepath.Join(t.scriptsDirectoryPath, "prepare_data.py"), "--data_directory_path", t.outputDataDirectoryPath)
	var b []byte
	astilog.Debugf("astiocr: executing <%s>", strings.Join(cmd.Args, " "))
	if b, err = cmd.CombinedOutput(); err != nil {
		err = errors.Wrapf(err, "astiocr: running %s failed with body %s", strings.Join(cmd.Args, " "), b)
		return
	}
	return
}
