# Installation

- [install tensorflow](https://www.tensorflow.org/install/)
- [install the tensorflow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- install `astiocr` 

```
$ go get -u github.com/asticode/go-astiocr/...
```

# Train your model

For the next commands we assume you are in `$GOPATH/src/github.com/asticode/go-astiocr`

## Set up configuration

Copy `astiocr/local.toml.dist` to `astiocr/local.toml` and replace the desired values.

## Gather data

Then run:

```
$ go run astiocr/main.go gather -v -c astiocr/local.toml
```

or if `make` is installed on your system:

```
$ make gather
```

## List available trained models

Run:

```
$ go run astiocr/main.go list -v -c astiocr/local.toml
```

or if `make` is installed on your system:

```
$ make list
```

## Configure the model

Run:

```
$ go run astiocr/main.go configure -v -c astiocr/local.toml -n <model name>
```

## Train the model

Move to your output path and run either `scripts/train.bat` or `scripts/train.sh` depending on your platform.
