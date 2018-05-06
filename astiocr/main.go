package main

import (
	"flag"

	"context"

	"github.com/asticode/go-astilog"
	"github.com/asticode/go-astiocr"
	"github.com/asticode/go-astitools/config"
	"github.com/asticode/go-astitools/flag"
	"github.com/asticode/go-astitools/os"
	"github.com/pkg/errors"
	"sort"
	"strings"
)

var configPath = flag.String("c", "", "the config path")
var name = flag.String("n", "", "the name")
var ctx, cancel = context.WithCancel(context.Background())

type Configuration struct {
	Trainer astiocr.ConfigurationTrainer `toml:"trainer"`
}

func main() {
	// Parse flags
	s := astiflag.Subcommand()
	flag.Parse()
	astilog.FlagInit()

	// Handle signals
	go astios.HandleSignals(astios.ContextSignalsFunc(cancel))

	// Parse configuration
	v, err := asticonfig.New(&Configuration{}, *configPath, &Configuration{})
	if err != nil {
		astilog.Fatal(errors.Wrap(err, "main: parsing configuration failed"))
	}
	c := v.(*Configuration)

	// Create trainer
	t, err := astiocr.NewTrainer(c.Trainer)
	if err != nil {
		astilog.Fatal(errors.Wrap(err, "main: creating trainer failed"))
	}

	// Switch on subcommand
	switch s {
	case "configure":
		// Check flag
		if len(*name) == 0 {
			astilog.Fatal("main: use -n to indicate a model name")
		}

		// Configure
		if err = t.Configure(ctx, *name); err != nil {
			astilog.Fatal(errors.Wrapf(err, "main: configuring model %s failed", *name))
		}
	case "gather":
		if err = t.Gather(ctx); err != nil {
			astilog.Fatal(errors.Wrap(err, "main: gathering failed"))
		}
	case "list":
		var m map[string]string
		if m, err = t.TrainedModels(ctx); err != nil {
			astilog.Fatal(errors.Wrap(err, "main: gathering failed"))
		}
		var models []string
		for n := range m {
			models = append(models, n)
		}
		sort.Strings(models)
		astilog.Infof("main: trained models are\n- %s", strings.Join(models, "\n- "))
	default:
		astilog.Fatal("main: no subcommand provided")
	}
}
