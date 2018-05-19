package astiocr

import (
	"context"
	"io/ioutil"
	"path/filepath"

	"github.com/pkg/errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// ConfigurationDetector represents a detector configuration
type ConfigurationDetector struct {
	// Path to the model
	ModelPath string `toml:"model_path"`
}

// Detector represents an object capable of detecting OCR
type Detector struct {
	g *tf.Graph
	s *tf.Session
}

// NewDetector creates a new detector
func NewDetector(c ConfigurationDetector) (d *Detector, err error) {
	// Init
	d = &Detector{}

	// Read the model
	var b []byte
	if b, err = ioutil.ReadFile(c.ModelPath); err != nil {
		err = errors.Wrapf(err, "astiocr: reading %s failed", c.ModelPath)
		return
	}

	// Create the graph
	d.g = tf.NewGraph()
	if err = d.g.Import(b, ""); err != nil {
		err = errors.Wrapf(err, "astiocr: importing model %s failed", c.ModelPath)
		return
	}

	// Create the session
	if d.s, err = tf.NewSession(d.g, nil); err != nil {
		err = errors.Wrap(err, "astiocr: creating session failed")
		return
	}
	return
}

// Close implements the io.Closer interface
func (d *Detector) Close() error {
	return d.s.Close()
}

// DetectionResult represents a detection result
type DetectionResult struct {
	Box         DetectionBox
	Label       string
	Probability float64
}

// DetectionBox represents a detection box
type DetectionBox struct {
	X1, X2 float64
	Y1, Y2 float64
}

// Detect detects OCR on an image
func (d *Detector) Detect(ctx context.Context, src string) (rs []DetectionResult, err error) {
	// Create tensor
	var t *tf.Tensor
	if t, err = d.tensorFromImage(src); err != nil {
		err = errors.Wrapf(err, "astiocr: creating tensor for image %s failed", src)
		return
	}

	// Run inference
	var probabilities, classes []float32
	var boxes [][]float32
	if probabilities, classes, boxes, err = d.runInference(t); err != nil {
		err = errors.Wrap(err, "astiocr: running inference failed")
		return
	}

	// Loop through results
	for idx := 0; idx < len(probabilities); idx++ {
		rs = append(rs, DetectionResult{
			Box: DetectionBox{
				X1: float64(boxes[idx][1]),
				X2: float64(boxes[idx][3]),
				Y1: float64(boxes[idx][0]),
				Y2: float64(boxes[idx][2]),
			},
			Label:       string(characters[int(classes[idx])-1]),
			Probability: float64(probabilities[idx]),
		})
	}
	return
}

func (d *Detector) tensorFromImage(src string) (t *tf.Tensor, err error) {
	// Read image
	var b []byte
	if b, err = ioutil.ReadFile(src); err != nil {
		err = errors.Wrapf(err, "astiocr: reading %s failed", src)
		return
	}

	// Create basic tensor
	if t, err = tf.NewTensor(string(b)); err != nil {
		err = errors.Wrap(err, "astiocr: creating basic tensor failed")
		return
	}

	// Create scope
	s := op.NewScope()
	input := op.Placeholder(s, tf.String)

	// Create output
	var o tf.Output
	switch filepath.Ext(src) {
	case ".jpg", ".jpeg":
		o = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	case ".png":
		o = op.DecodePng(s, input, op.DecodePngChannels(3))
	default:
		o = op.DecodeBmp(s, input, op.DecodeBmpChannels(3))
	}
	output := op.ExpandDims(
		s,
		o,
		op.Const(s.SubScope("make_batch"), int32(0)),
	)

	// Create graph
	var graph *tf.Graph
	if graph, err = s.Finalize(); err != nil {
		err = errors.Wrap(err, "astiocr: finalizing failed")
		return
	}

	// Create session
	var sess *tf.Session
	if sess, err = tf.NewSession(graph, nil); err != nil {
		err = errors.Wrap(err, "astiocr: creating session failed")
		return
	}
	defer sess.Close()

	// Normalize
	var ts []*tf.Tensor
	if ts, err = sess.Run(
		map[tf.Output]*tf.Tensor{input: t},
		[]tf.Output{output},
		nil,
	); err != nil {
		err = errors.Wrap(err, "astiocr: normalizing failed")
		return
	}
	t = ts[0]
	return
}

func (d *Detector) runInference(t *tf.Tensor) (probabilities, classes []float32, boxes [][]float32, err error) {
	// Input
	i := d.g.Operation("image_tensor")

	// Outputs
	o1 := d.g.Operation("detection_boxes")
	o2 := d.g.Operation("detection_scores")
	o3 := d.g.Operation("detection_classes")
	o4 := d.g.Operation("num_detections")

	// Run
	var os []*tf.Tensor
	if os, err = d.s.Run(
		map[tf.Output]*tf.Tensor{i.Output(0): t},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil,
	); err != nil {
		err = errors.Wrap(err, "astiocr: running session failed")
		return
	}

	// Get results
	probabilities = os[1].Value().([][]float32)[0]
	classes = os[2].Value().([][]float32)[0]
	boxes = os[0].Value().([][][]float32)[0]
	return
}
