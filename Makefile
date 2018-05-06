configure:
	go run astiocr/main.go configure -v -c astiocr/local.toml -n ssd_mobilenet_v2_coco

gather:
	go run astiocr/main.go gather -v -c astiocr/local.toml

list:
	go run astiocr/main.go list -v -c astiocr/local.toml