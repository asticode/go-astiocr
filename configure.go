package astiocr

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/asticode/go-astilog"
	"github.com/asticode/go-astitools/archive"
	"github.com/asticode/go-astitools/http"
	"github.com/asticode/go-astitools/os"
	"github.com/pkg/errors"
)

const numSteps = "10000"

// ^\[([^\[]+)\]\((.+)\)
var regexpTrainedModel = regexp.MustCompile("^\\[([^\\[]+)\\]\\((.+)\\)")

// List lists available trained models
func (t *Trainer) TrainedModels(ctx context.Context) (models map[string]string, err error) {
	// Open file
	var f *os.File
	p := filepath.Join(t.tensorFlowModelsDirectoryPath, "research", "object_detection", "g3doc", "detection_model_zoo.md")
	if f, err = os.Open(p); err != nil {
		err = errors.Wrapf(err, "astiocr: opening %s failed", p)
		return
	}
	defer f.Close()

	// Create reader
	r := bufio.NewReader(f)

	// Loop through lines
	var inModelSection bool
	models = make(map[string]string)
	for {
		// Get next line
		var l string
		if l, err = r.ReadString('\n'); err != nil {
			if err == io.EOF {
				err = nil
				break
			}
			err = errors.Wrapf(err, "astiocr: reading line of %s failed", p)
			return
		}

		// This title indicates a model section
		if strings.HasPrefix(l, "## ") && strings.Index(l, "trained models") > -1 {
			inModelSection = true
			continue
		}

		// Not in a model section
		if !inModelSection {
			continue
		}

		// Remove table row markdown styling
		l = strings.TrimPrefix(l, "| ")

		// Apply regexp
		matches := regexpTrainedModel.FindAllStringSubmatch(l, -1)
		if len(matches) > 0 && len(matches[0]) >= 3 {
			models[matches[0][1]] = matches[0][2]
		}
	}
	return
}

// Configure configures the model
func (t *Trainer) Configure(ctx context.Context, modelName string) (err error) {
	// Create configure folders
	if err = t.createConfigureFolders(); err != nil {
		err = errors.Wrap(err, "astiocr: creating configure folders failed")
		return
	}

	// Get trained models
	var trainedModels map[string]string
	if trainedModels, err = t.TrainedModels(ctx); err != nil {
		err = errors.Wrap(err, "astiocr: getting trained models failed")
		return
	}

	// Requested model doesn't exist
	url, ok := trainedModels[modelName]
	if !ok {
		err = fmt.Errorf("astiocr: model %s doesn't exist", modelName)
		return
	}

	// Create train scripts
	if err = t.createTrainScripts(ctx); err != nil {
		err = errors.Wrap(err, "astiocr: copying train script failed")
		return
	}

	// Create config file
	if err = t.createConfigFile(ctx, modelName); err != nil {
		err = errors.Wrap(err, "astiocr: creating config file failed")
		return
	}

	// Set up trained model
	if err = t.setUpTrainedModel(ctx, url); err != nil {
		err = errors.Wrapf(err, "astiocr: setting up trained model %s failed", modelName)
		return
	}
	return
}

func (t *Trainer) createConfigureFolders() (err error) {
	// Remove folders
	for _, p := range []string{
		t.outputConfigDirectoryPath,
		t.outputOutputDirectoryPath,
		t.outputScriptsDirectoryPath,
	} {
		astilog.Debugf("astiocr: removing %s", p)
		if err = os.RemoveAll(p); err != nil {
			err = errors.Wrapf(err, "astiocr: removeAll %s failed", p)
			return
		}
	}

	// Loop through folders to create
	for _, p := range []string{
		t.cacheDirectoryPath,
		t.outputConfigDirectoryPath,
		t.outputOutputDirectoryPath,
		filepath.Join(t.outputOutputDirectoryPath, "eval"),
		filepath.Join(t.outputOutputDirectoryPath, "model"),
		filepath.Join(t.outputOutputDirectoryPath, "training"),
		t.outputScriptsDirectoryPath,
	} {
		astilog.Debugf("astiocr: creating %s", p)
		if err = os.MkdirAll(p, 0700); err != nil {
			err = errors.Wrapf(err, "astiocr: mkdirall %s failed", p)
		}
	}
	return
}

var trainScript = "python scripts/train.py --logtostderr --train_dir=output/training --pipeline_config_path=config/model.config"
var evalScript = "python3 scripts/eval.py --logtostderr --checkpoint_dir=output/training --pipeline_config_path=config/model.config --eval_dir=output/eval"
var exportInferenceGraphScript = "python scripts/export_inference_graph.py --input_type image_tensor --pipeline_config_path=config/model.config --trained_checkpoint_prefix output/training/model.ckpt-" + numSteps + " --output_directory output/model"

func (t *Trainer) createTrainScripts(ctx context.Context) (err error) {
	// Copy files
	for _, n := range []string{
		"train",
		"eval",
		"export_inference_graph",
	} {
		src := filepath.Join(t.tensorFlowModelsDirectoryPath, "research", "object_detection", n+".py")
		dst := filepath.Join(t.outputScriptsDirectoryPath, n+".py")
		astilog.Debugf("astiocr: copying %s to %s", src, dst)
		if err = astios.Copy(ctx, src, dst); err != nil {
			err = errors.Wrap(err, "astiocr: copying train script failed")
			return
		}
	}

	// Create scripts
	for n, s := range map[string]string{
		"train":  trainScript,
		"eval":   evalScript,
		"export": exportInferenceGraphScript,
	} {
		for _, ext := range []string{
			".bat",
			".sh",
		} {
			p := filepath.Join(t.outputScriptsDirectoryPath, n+ext)
			astilog.Debugf("astiocr: creating %s script in %s", n, p)
			if err = t.createScript(ctx, p, s); err != nil {
				err = errors.Wrapf(err, "astiocr: creating %s script in %s failed", n, p)
				return
			}
		}
	}
	return
}

func (t *Trainer) createScript(ctx context.Context, p, s string) (err error) {
	// Create file
	var f *os.File
	if f, err = os.OpenFile(p, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0700); err != nil {
		err = errors.Wrapf(err, "astiocr: creating %s failed", p)
		return
	}
	defer f.Close()

	// Write
	if _, err = f.WriteString(s); err != nil {
		err = errors.Wrapf(err, "astiocr: writing to %s failed", p)
		return
	}
	return
}

// Regexps
var (
	regexpBatchSize          = regexp.MustCompile("batch_size\\: ([\\d]+)")
	regexpFineTuneCheckpoint = regexp.MustCompile("fine_tune_checkpoint\\: (\".+\")")
	regexpInputPath          = regexp.MustCompile("input_path\\: (\".+\")")
	regexpLabelMapPath       = regexp.MustCompile("label_map_path\\: (\".+\")")
	regexpNumClasses         = regexp.MustCompile("num_classes\\: ([\\d]+)")
	regexpNumExamples        = regexp.MustCompile("num_examples\\: ([\\d]+)")
	regexpNumSteps           = regexp.MustCompile("num_steps\\: ([\\d]+)")
)

func (t *Trainer) createConfigFile(ctx context.Context, modelName string) (err error) {
	// Open file
	src := filepath.Join(t.tensorFlowModelsDirectoryPath, "research", "object_detection", "samples", "configs", modelName+".config")
	astilog.Debugf("astiocr: opening %s", src)
	var srcFile *os.File
	if srcFile, err = os.Open(src); err != nil {
		err = errors.Wrapf(err, "astiocr: opening %s failed", src)
		return
	}
	defer srcFile.Close()

	// Create file
	dst := filepath.Join(t.outputConfigDirectoryPath, "model.config")
	astilog.Debugf("astiocr: creating %s", dst)
	var dstFile *os.File
	if dstFile, err = os.Create(dst); err != nil {
		err = errors.Wrapf(err, "astiocr: creating %s failed", dst)
		return
	}
	defer dstFile.Close()

	// Create reader
	r := bufio.NewReader(srcFile)

	// Loop through lines
	astilog.Debugf("astiocr: updating %s", dst)
	var inEvalReader bool
	for {
		// Read line
		var l string
		var errRead error
		if l, errRead = r.ReadString('\n'); errRead != nil && errRead != io.EOF {
			err = errors.Wrap(err, "astiocr: reading line failed")
			return
		}

		// Update reader placement
		if strings.HasPrefix(l, "eval_input_reader") {
			inEvalReader = true
		}

		// Loop through regexp
		for r, v := range map[*regexp.Regexp]string{
			regexpNumClasses:         "2", // strconv.Itoa(len(characters)),
			regexpBatchSize:          "10",
			regexpFineTuneCheckpoint: "\"config/model.ckpt\"",
			regexpNumSteps:           numSteps,
			regexpNumExamples:        strconv.Itoa(t.testDataCount),
			regexpInputPath:          "",
			regexpLabelMapPath:       "\"data/label_map.pbtxt\"",
		} {
			// Get matches
			matches := r.FindAllStringSubmatchIndex(l, -1)
			if len(matches) <= 0 || len(matches[0]) < 4 {
				continue
			}

			// Special value
			if r == regexpInputPath {
				v = "\"data/"
				if inEvalReader {
					v += "test"
				} else {
					v += "training"
				}
				v += "/data.record\""
			}

			// Replace
			l = l[:matches[0][2]] + v + l[matches[0][3]:]
			break
		}

		// Write line
		if _, err = dstFile.WriteString(l); err != nil {
			err = errors.Wrapf(err, "astiocr: writing %s to %s failed", l, dst)
			return
		}

		// Check EOF
		if errRead == io.EOF {
			break
		}
	}
	return
}

func (t *Trainer) setUpTrainedModel(ctx context.Context, url string) (err error) {
	// Create temp dir
	var tempDirPath string
	if tempDirPath, err = ioutil.TempDir(os.TempDir(), "astiocr_trainer_"); err != nil {
		err = errors.Wrap(err, "astiocr: creating temp dir failed")
		return
	}
	astilog.Debugf("astiocr: created temp dir %s", tempDirPath)

	// Make sure to remove the temp dir at the end of the process
	defer func() {
		// Remove
		astilog.Debugf("astiocr: removing %s", tempDirPath)
		if errDefer := os.RemoveAll(tempDirPath); errDefer != nil {
			astilog.Error(errors.Wrapf(errDefer, "astiocr: removing %s failed", tempDirPath))
			return
		}
	}()

	// Download
	p := filepath.Join(t.cacheDirectoryPath, filepath.Base(url))
	if _, err = os.Stat(p); err != nil && !os.IsNotExist(err) {
		err = errors.Wrapf(err, "astiocr: stating %s failed", p)
	} else if os.IsNotExist(err) {
		astilog.Debugf("astiocr: downloading %s to %s", url, p)
		if err = astihttp.Download(ctx, &http.Client{}, url, p); err != nil {
			err = errors.Wrapf(err, "astiocr: downloading %s to %s failed", url, p)
			return
		}
	} else {
		astilog.Debugf("astiocr: %s already exists, skipping download of %s", p, url)
	}

	// Untar
	astilog.Debugf("astiocr: untaring %s into %s", p, tempDirPath)
	if err = astiarchive.Untar(ctx, p, tempDirPath); err != nil {
		err = errors.Wrapf(err, "astiocr: untaring %s into %s failed", p, tempDirPath)
		return
	}

	// Walk through temp dir
	if err = filepath.Walk(tempDirPath, func(path string, info os.FileInfo, e error) (err error) {
		// Process error
		if e != nil {
			return e
		}

		// Only process files which contain ".ckpt."
		b := filepath.Base(info.Name())
		if info.IsDir() || strings.Index(b, ".ckpt.") == -1 {
			return
		}

		// Copy file
		dst := filepath.Join(t.outputConfigDirectoryPath, b)
		astilog.Debugf("astiocr: copying %s to %s", path, dst)
		if err = astios.Copy(ctx, path, dst); err != nil {
			err = errors.Wrapf(err, "astiocr: copying %s into %s failed", path, dst)
			return
		}
		return
	}); err != nil {
		err = errors.Wrapf(err, "astiocr: walking through %s failed", tempDirPath)
		return
	}
	return
}
