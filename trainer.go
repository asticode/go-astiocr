package astiocr

import (
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/asticode/go-astitools/image"
	"github.com/golang/freetype/truetype"
	"github.com/pkg/errors"
	"golang.org/x/image/font/gofont/gomono"
)

// ConfigurationTrainer represents a trainer configuration
type ConfigurationTrainer struct {
	// Path to the cache directory
	CacheDirectoryPath string `toml:"cache_directory_path"`

	// Number of images generated for both training and test purposes
	Count int `toml:"count"`

	// Color options
	Colors []ConfigurationColor `toml:"colors"`

	// Font options
	Fonts []ConfigurationFont `toml:"fonts"`

	// Image options
	Image ConfigurationImage `toml:"image"`

	// Path to the output directory
	OutputDirectoryPath string `toml:"output_directory_path"`

	// Path to the python binary
	PythonBinaryPath string `toml:"python_binary_path"`

	// Path to the scripts directory
	ScriptsDirectoryPath string `toml:"scripts_directory_path"`

	// Show box around labels
	ShowBox bool `toml:"show_box"`

	// Show entire grid
	ShowGrid bool `toml:"show_grid"`

	// Path to the tensorflow models directory
	TensorFlowModelsDirectoryPath string `toml:"tensorflow_models_directory_path"`

	// The proportion of test data in the generated images
	TestDataProportion float64 `toml:"test_data_proportion"`
}

// ConfigurationColor represents a color configuration
type ConfigurationColor struct {
	Background astiimage.RGBA   `toml:"background"`
	Fonts      []astiimage.RGBA `toml:"fonts"`
}

// ConfigurationFont represents a font configuration
type ConfigurationFont struct {
	File          string  `toml:"file"`
	PositionRatio float64 `toml:"position_ratio"`
}

// ConfigurationImage represents an image configuration
type ConfigurationImage struct {
	Height int `toml:"height"`
	Width  int `toml:"width"`
}

// Trainer represents an object capable of training a model
type Trainer struct {
	cacheDirectoryPath            string
	count                         int
	colors                        []ConfigurationColor
	fonts                         []*font
	image                         ConfigurationImage
	outputConfigDirectoryPath     string
	outputDataDirectoryPath       string
	outputDirectoryPath           string
	outputOutputDirectoryPath      string
	outputScriptsDirectoryPath    string
	pythonBinaryPath              string
	scriptsDirectoryPath          string
	showBox                       bool
	showGrid                      bool
	tensorFlowModelsDirectoryPath string
	testDataCount                 int
	testDataProportion            float64
	trainingDataCount             int
}

type font struct {
	body          []byte
	font          *truetype.Font
	name          string
	positionRatio float64
}

// NewTrainer creates a new trainer
func NewTrainer(c ConfigurationTrainer) (t *Trainer, err error) {
	// Init
	t = &Trainer{
		showBox:                       c.ShowBox,
		showGrid:                      c.ShowGrid,
		tensorFlowModelsDirectoryPath: c.TensorFlowModelsDirectoryPath,
	}

	// Count
	t.count = c.Count
	if t.count == 0 {
		t.count = 1
	}

	// Test data proportion
	t.testDataProportion = c.TestDataProportion
	if t.testDataProportion == 0 {
		t.testDataProportion = 30.0
	}

	// Training and test data counts
	t.testDataCount = int(t.testDataProportion * float64(t.count) / 100)
	t.trainingDataCount = t.count - t.testDataCount
	if t.testDataCount == 0 {
		t.testDataCount++
		t.count++
	}

	// Colors
	t.colors = c.Colors
	if len(t.colors) == 0 {
		t.colors = []ConfigurationColor{{
			Background: *astiimage.NewRGBA(0xff, 0, 0, 0),
			Fonts:      []astiimage.RGBA{*astiimage.NewRGBA(0xff, 0xff, 0xff, 0xff)},
		}}
	}

	// Loop through fonts
	if len(c.Fonts) > 0 {
		// Read files
		for _, f := range c.Fonts {
			var b []byte
			if b, err = ioutil.ReadFile(f.File); err != nil {
				err = errors.Wrapf(err, "astiocr: reading file %s failed", f.File)
				return
			}
			nft := &font{
				body:          b,
				name:          f.File,
				positionRatio: f.PositionRatio,
			}
			if nft.positionRatio == 0 {
				nft.positionRatio = 2.5
			}
			t.fonts = append(t.fonts, nft)
		}
	} else {
		t.fonts = append(t.fonts, &font{
			body:          gomono.TTF,
			name:          "gomono",
			positionRatio: 2.5,
		})
	}

	// Parse fonts
	for _, f := range t.fonts {
		if f.font, err = truetype.Parse(f.body); err != nil {
			err = errors.Wrapf(err, "astiocr: parsing font %s failed", f.name)
			return
		}
	}

	// Image
	t.image.Height = c.Image.Height
	t.image.Width = c.Image.Width
	if t.image.Height == 0 {
		t.image.Height = 360
	}
	if t.image.Width == 0 {
		t.image.Width = 640
	}

	// Get current directory path
	var cd string
	if cd, err = os.Getwd(); err != nil {
		err = errors.Wrap(err, "astiocr: getting working directory failed")
		return
	}

	// Output directory path
	t.outputDirectoryPath = c.OutputDirectoryPath
	if len(t.outputDirectoryPath) == 0 {
		t.outputDirectoryPath = filepath.Join(cd, "tmp")
	}
	t.outputConfigDirectoryPath = filepath.Join(t.outputDirectoryPath, "config")
	t.outputDataDirectoryPath = filepath.Join(t.outputDirectoryPath, "data")
	t.outputOutputDirectoryPath = filepath.Join(t.outputDirectoryPath, "output")
	t.outputScriptsDirectoryPath = filepath.Join(t.outputDirectoryPath, "scripts")

	// Python binary path
	t.pythonBinaryPath = c.PythonBinaryPath
	if len(t.pythonBinaryPath) == 0 {
		t.pythonBinaryPath = "python3"
	}

	// Scripts directory path
	t.scriptsDirectoryPath = c.ScriptsDirectoryPath
	if len(t.scriptsDirectoryPath) == 0 {
		t.scriptsDirectoryPath = filepath.Join(cd, "scripts")
	}

	// Cache directory path
	t.cacheDirectoryPath = c.CacheDirectoryPath
	if len(t.cacheDirectoryPath) == 0 {
		t.cacheDirectoryPath = filepath.Join(os.TempDir(), "astiocr_cache")
	}
	return
}
