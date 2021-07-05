MAKEFILE_NAME := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
APP_ROOT      := $(abspath $(lastword $(MAKEFILE_NAME))/..)
APP_NAME	  := CRIM LibTorch Extensions (C++/Python)
CMAKE 		  ?= $(shell which cmake3 || which cmake)


BUILD_DIR ?= $(APP_ROOT)/build
ifeq ($(BUILD_DIR),)
  BUILD_DIR := $(APP_ROOT)/build
endif
INSTALL_DIR ?= $(APP_ROOT)/install
ifeq ($(INSTALL_DIR),)
  INSTALL_DIR := $(APP_ROOT)/install
endif

# remove trailing slash and spaces
BUILD_DIR := $(shell realpath $(dir $(BUILD_DIR)/))
INSTALL_DIR := $(shell realpath $(dir $(INSTALL_DIR)/))

## --- Information targets --- ##

.DEFAULT_GOAL := help
all: help

# Auto documented help targets & sections from comments
#	- detects lines marked by double octothorpe (#), then applies the corresponding target/section markup
#   - target comments must be defined after their dependencies (if any)
#	- section comments must have at least a double dash (-)
#
# 	Original Reference:
#		https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# 	Formats:
#		https://misc.flogisoft.com/bash/tip_colors_and_formatting
_SECTION := \033[34m
_TARGET  := \033[36m
_NORMAL  := \033[0m
_SPACING := 24
.PHONY: help
# note: use "\#\#" to escape results that would self-match in this target's search definition
help:	## print this help message (default)
	@echo "$(_SECTION)=== $(APP_NAME) help ===$(_NORMAL)"
	@echo "Please use 'make <target>' where <target> is one of:"
#	@grep -E '^[a-zA-Z_-]+:.*?\#\# .*$$' $(MAKEFILE_LIST) \
#		| awk 'BEGIN {FS = ":.*?\#\# "}; {printf "    $(_TARGET)%-24s$(_NORMAL) %s\n", $$1, $$2}'
	@grep -E '\#\#.*$$' "$(APP_ROOT)/$(MAKEFILE_NAME)" \
		| awk ' BEGIN {FS = "(:|\-\-\-)+.*?\#\# "}; \
			/\--/ 		{printf "$(_SECTION)%s$(_NORMAL)\n", $$1;} \
			/:/   		{printf "   $(_TARGET)%-$(_SPACING)s$(_NORMAL) %s\n", $$1, $$2;} \
			/\-only:/   {gsub(/-only/, "", $$1); \
						 printf "   $(_TARGET)%-$(_SPACING)s$(_NORMAL) %s (preinstall dependencies)\n", $$1, $$2;} \
		'

.PHONY: info
info:  ## Display useful information about configurations employed by make
	@echo "APP_NAME:       $(APP_NAME)"
	@echo "APP_ROOT:       $(APP_ROOT)"
	@echo "CMAKE:          $(CMAKE)"
	@echo "BUILD_DIR:      $(BUILD_DIR)"
	@echo "INSTALL_DIR:    $(INSTALL_DIR)"

## --- Cleanup targets --- ##

.PHONY: clean
clean: clean-build clean-install  ## clean everything

.PHONY: clean-build
clean-build:	## clean build caches
	@-rm -fr "$(BUILD_DIR)/"
	@-rm -fr "$(APP_ROOT)/CMakeFiles/"
	@-rm -fr "$(APP_ROOT)/CMakeCache.txt"

.PHONY: clean-install
clean-install:	## clean output install locations
	@-rm -fr *.egg
	@-rm -fr *.egg-info
	@-rm -fr "$(APP_ROOT)/bin"
	@-rm -fr "$(APP_ROOT)/bdist"
	@-rm -fr "$(APP_ROOT)/dist"
	@-rm -fr "$(APP_ROOT)/install"
	@-rm -fr "$(BUILD_DIR)/bin*"
	@-rm -fr "$(BUILD_DIR)/bdist*"
	@-rm -fr "$(BUILD_DIR)/dist*"
	@-rm -fr "$(BUILD_DIR)/lib*"
	@-rm -fr "$(BUILD_DIR)/install*"

## --- Project targets --- ##

.PHONY: build
build:	## build C++ library extensions from source (refer to CMake variables to configure build)
	@mkdir -p "$(BUILD_DIR)"
	@cd "$(BUILD_DIR)" && \
		$(CMAKE) \
			-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
			-DTORCH_DIR=${TORCH_DIR} \
			-DTORCHVISION_DIR=${TORCHVISION_DIR} \
			-DOPENCV_DIR=${OPENCV_DIR} \
			-DCLI11_DIR=${CLI11_DIR} \
		"$(APP_ROOT)"
	@cd "$(BUILD_DIR)" && make

.PHONY: install
install: install-cpp  ## alias to 'install-cpp'

.PHONY: install-cpp
install-cpp: build  ## install built C++ libraries
	@cd "$(BUILD_DIR)" && make install

.PHONY: install-python
install-python:  ## install library extension with Python/C++ bindings into the current Python environment
	python setup.py install

.PHONY: test-bench-help
test-bench-help:  ## call the help of the TestBench application (attempts building it if it doesn't exist)
	@test -f "$(BUILD_DIR)/TestBench/TestBench" || ( \
		echo "TestBench was not found. Attempting to build and install it." && \
		$(MAKE) -j $(shell nproc) build \
	)
	@bash -c '$(BUILD_DIR)/TestBench/TestBench --help'

## --- Versioning targets --- ##

# Bumpversion 'dry' config
# if 'dry' is specified as target, any bumpversion call using 'BUMP_XARGS' will not apply changes
BUMP_XARGS ?= --verbose --allow-dirty
ifeq ($(filter dry, $(MAKECMDGOALS)), dry)
  BUMP_XARGS := $(BUMP_XARGS) --dry-run
endif

.PHONY: dry
dry: setup.cfg	## run 'bump' target without applying changes (dry-run)
ifeq ($(findstring bump, $(MAKECMDGOALS)),)
	$(error Target 'dry' must be combined with a 'bump' target)
endif

.PHONY: bump
bump:	## bump version using VERSION specified as user input (make VERSION=<X.Y.Z> bump)
	@-echo "Updating package version ..."
	@[ "${VERSION}" ] || ( echo ">> 'VERSION' is not set"; exit 1 )
	@-bash -c '$(CONDA_CMD) test -f "$(CONDA_ENV_PATH)/bin/bump2version" || pip install $(PIP_XARGS) bump2version'
	@-bash -c '$(CONDA_CMD) bump2version $(BUMP_XARGS) --new-version "${VERSION}" patch;'
