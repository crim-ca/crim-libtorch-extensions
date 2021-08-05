MAKEFILE_NAME := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
APP_ROOT      := $(abspath $(lastword $(MAKEFILE_NAME))/..)
APP_NAME	  := CRIM LibTorch Extensions (C++/Python)
MAKEFILE_PATH := $(APP_ROOT)/$(MAKEFILE_NAME)
CMAKE 		  ?= $(shell which cmake3 || which cmake)	## cmake binary (path) to employ for building the project
DEBUG		  ?=
ifneq ($(DEBUG),)
  CMAKE_DEBUG := --log-level=DEBUG
endif

# NOTE: (comments)
# 	All *literal* comments should only be preceded by a single '#'
#	Double '#' entries will be captured as potential auto-doc target or option.
#	Triple '#' entries will be captured as sections under which targets are listed.

# NOTE: (options)
#	All 'options' that provide configurable inputs should be followed by a corresponding call using:
#		VAR := $(call clean_opt,$(VAR))
#	Using 'clean_opt' operation avoids unexpected spaces because of leading comments (be careful, no space after comma).
#	Do not use make's 'strip' command as it removes *all* duplicate spaces, including within the option string itself.
# 	Use tabs after the default value of options to define the comment to make display cleaner.

clean_opt = $(shell echo "$(1)" | sed -r -e "s/[ '$'\t'']+$$//g")


# color codes and utility shortcuts
COLOR  ?= 1		## enable or disable use of colors in outputs (eg: 'COLOR=0' will disable them)
COLOR  := $(call clean_opt,$(COLOR))
ECHO   := echo
# all colors are undefined by default
# only define items that have actual text on top of colors
# single '#' added at end ensure messages get aligned after level name
_ERROR := ERROR #
_WARN  := WARN  #
_INFO  := INFO  #
_DEBUG := DEBUG #
_MAKE  := "make"
ifneq ("$(COLOR)", "0")
  # must make sure that interpretation of backslash escapes is enabled (not default for all terminals)
  # must use the full path otherwise make confuses it with its own echo command
  # reference: https://misc.flogisoft.com/bash/tip_colors_and_formatting
  ECHO      		:= /bin/echo -e
  _ESC      		:= $(shell printf '\033')
  _NORMAL   		:= $(_ESC)[0m
  _RED      		:= $(_ESC)[31m
  _GREEN    		:= $(_ESC)[32m
  _YELLOW   		:= $(_ESC)[33m
  _BLUE     		:= $(_ESC)[34m
  _MAGENTA  		:= $(_ESC)[35m
  _CYAN     		:= $(_ESC)[36m
  _DARK_GRAY 		:= $(_ESC)[90m
  _LIGHT_CYAN 		:= $(_ESC)[96m
  _DARK_PURPLE 		:= $(_ESC)[38;5;60m
  _BG_GRAY  		:= $(_ESC)[100m
  _BG_DARK_RED 		:= $(_ESC)[48;5;52m
  _BG_PURPLE  		:= $(_ESC)[48;5;55m
  _BG_LIGHT_PURPLE 	:= $(_ESC)[48;5;60m
  _ORANGE   		:= $(_ESC)[38;5;166m
  # concepts
  _ERROR    		:= $(_RED)$(_ERROR)$(_NORMAL)
  _WARN     		:= $(_YELLOW)$(_WARN)$(_NORMAL)
  _INFO     		:= $(_BLUE)$(_INFO)$(_NORMAL)
  _DEBUG    		:= $(_DARK_GRAY)$(_DEBUG)$(_NORMAL)
  _MAKE     		:= $(_ORANGE)$(_MAKE)$(_NORMAL)
  _FILE     		:= $(_LIGHT_CYAN)
  _VAR				:= $(_MAGENTA)
  _HEADER   		:= $(_BG_DARK_RED)
  _SECTION  		:= $(_BG_GRAY)
  _TARGET   		:= $(_CYAN)
  _OPTION   		:= $(_MAGENTA)
  _CONFIG   		:= $(_GREEN)
else
  # reuse for Singularity
  NO_COLOR := --nocolor
endif

# build/install paths
BUILD_DIR ?= $(APP_ROOT)/build
ifeq ($(BUILD_DIR),)
  BUILD_DIR := $(APP_ROOT)/build
endif
INSTALL_DIR ?= $(APP_ROOT)/install
ifeq ($(INSTALL_DIR),)
  INSTALL_DIR := $(APP_ROOT)/install
endif
PYTHON_EXECUTABLE ?= $(shell realpath $${PYTHON_EXECUTABLE} 2>/dev/null || which python)
ifeq ($(PYTHON_ROOT_DIR),)
  PYTHON_ROOT_DIR := $(shell dirname $(PYTHON_EXECUTABLE))
  PYTHON_ROOT_DIR := $(shell dirname $(PYTHON_ROOT_DIR))
endif

# remove trailing slash and spaces
BUILD_DIR := $(shell realpath $(dir $(BUILD_DIR)/))
INSTALL_DIR := $(shell realpath $(dir $(INSTALL_DIR)/))
PYTHON_ROOT_DIR := $(shell realpath $(dir $(PYTHON_ROOT_DIR)/))
APP_CONFIG_VARS := \
	APP_NAME \
	APP_ROOT \
	CMAKE \
	BUILD_DIR \
	INSTALL_DIR \
	PYTHON_EXECUTABLE \
	PYTHON_ROOT_DIR

# build vars (from env or override by argument)
TORCH_DIR 		?= $(shell echo $${TORCH_DIR})
TORCHVISION_DIR ?= $(shell echo $${TORCHVISION_DIR})
OPENCV_DIR 		?= $(shell echo $${OPENCV_DIR})
CLI11_DIR 		?= $(shell echo $${CLI11_DIR})
PLOG_DIR 		?= $(shell echo $${PLOG_DIR})
BUILD_CONFIG_VARS := \
	TORCH_DIR \
	TORCHVISION_DIR \
	OPENCV_DIR \
	CLI11_DIR \
	PLOG_DIR

# demo config (use 'test-bench-help' target for all option details)
# use by default known ImageNet location/data type
DEMO_DATA_ROOT_DIR 	?= /misc/data20/visi/imagenet
DEMO_DATA_EXT 		?= jpeg
DEMO_MODEL 			?= EfficientNetB0
DEMO_OPTIM 			?= SGD
DEMO_MAX_EPOCHS 	?= 30
DEMO_BATCH_SIZE 	?= 16
DEMO_WORKERS		?= -1
DEMO_LOG_LEVEL		?= debug
DEMO_LOG_FILE       ?= $(APP_ROOT)/TestBench.log
DEMO_XARGS          ?=
DEMO_CONFIG_VARS := \
	DEMO_DATA_ROOT_DIR \
	DEMO_DATA_EXT \
	DEMO_MODEL \
	DEMO_OPTIM \
	DEMO_MAX_EPOCHS \
	DEMO_BATCH_SIZE \
	DEMO_WORKERS \
	DEMO_LOG_LEVEL \
	DEMO_LOG_FILE \
	DEMO_XARGS

### --- Information targets --- ###

.DEFAULT_GOAL := help
all: help

.PHONY: help
help: targets

# Auto documented help targets & sections from comments
#	- detects lines marked by double octothorpe (#), then applies the corresponding target/section markup
#   - target comments must be defined after their dependencies (if any)
#	- section comments must have at least a double dash (-)
#
# 	Original Reference:
#		https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# 	Formats:
#		https://misc.flogisoft.com/bash/tip_colors_and_formatting
_TARGET_INDENT := 24
.PHONY: targets
# note: use "\#\#" to escape results that would self-match in this target's search definition
targets:	## print available targets
	@$(ECHO) "$(_HEADER)=== $(APP_NAME) help ===$(_NORMAL)"
	@$(ECHO) "Use '$(_MAKE) [$(_OPTION)OPTION$(_NORMAL)=<value>] <$(_TARGET)target$(_NORMAL)>'"
	@$(ECHO) "where <$(_TARGET)target$(_NORMAL)> is one of:"
	@$(ECHO) ""
	@grep -v -E '.*(\?|\:\=).*$$' "$(MAKEFILE_PATH)" | grep -E '\#\#.*$$' \
			| sed -r -e 's/([\sA-Za-z0-9\_\-]+)\:[\sA-Za-z0-9\_\-\.]+\s*\#{2,3}\s*(.*)\s*$$/\1!!!\2/g' \
			| awk ' BEGIN {FS = "(:|###)+.*?## "}; \
				/\#\#\#/ 	{printf "$(_SECTION)%s$(_NORMAL)\n", $$1;} \
				/:/   	 	{printf "  $(_TARGET)%-$(_TARGET_INDENT)s$(_NORMAL) %s\n", $$1, $$2;} \
		 	  '

TARGETS := $(shell \
			 grep -E '^[A-Za-z0-9\_\-]+\:.*$$' "$(MAKEFILE_PATH)" \
			 | sed -r -e 's/([A-Za-z0-9\_\-]+)\:.*$$/\1/g')

.PHONY: target-names
target-names: 	## display literal listing of all detected targets names
	@for target in $(sort $(TARGETS)); do $(ECHO) "$${target}"; done

# notes:
#	Match all lines that have an override-able option ('?' instead of ':' before '=').
#	Split the matched lines into variable name, default value and description comment (if any).
#	Display results by aligned columns and with colors (if enabled).
.PHONY: options
options:     ## display configurable option details for execution of targets
	@$(ECHO) "$(_HEADER)=== Configurable Options [$(APP_NAME)] ===$(_NORMAL)\n"
	@$(ECHO) "$(_INFO)Use '$(_MAKE) $(_OPTION)OPTION$(_NORMAL)=<value> <$(_TARGET)target$(_NORMAL)>'"
	@$(ECHO) "where <$(_OPTION)OPTION$(_NORMAL)> is one amongst below choices:\n"
	@grep -E '.*\?\=.*$$' "$(MAKEFILE_PATH)" \
 		| sed -r -e 's/\s*(.*)\s*\?\=[\s\t]*([^#]*[^#\s\t]+)?[\s\t]*(##\s+(.*))*$$/\1!!!\2!!!\3!!!\4/g' \
 		| awk ' BEGIN {FS = "!!!"}; {printf "  $(_MAGENTA)%-24s$(_NORMAL)%s%-27s[default:%s]\n", $$1, $$4, "\n", $$2}'
	@$(ECHO) ""

_INFO_INDENT := 24
.PHONY: info
info:  ## Display useful information about configurations employed by make
	@$(ECHO) "$(_SECTION)Build/Install Configuration$(_NORMAL)"
	@$(foreach VAR,$(APP_CONFIG_VARS),\
		printf "  $(_VAR)%-$(_INFO_INDENT)s$(_NORMAL)%s\n" $(VAR) "$($(VAR))";\
	)
	@$(ECHO) "$(_SECTION)Build Dependencies$(_NORMAL)"
	@$(foreach VAR,$(BUILD_CONFIG_VARS),\
		printf "  $(_VAR)%-$(_INFO_INDENT)s$(_NORMAL)%s\n" $(VAR) "$($(VAR))";\
	)
	@$(ECHO) "$(_SECTION)Demo/Test Configuration$(_NORMAL)"
	@$(foreach VAR,$(DEMO_CONFIG_VARS),\
		printf "  $(_VAR)%-$(_INFO_INDENT)s$(_NORMAL)%s\n" $(VAR) "$($(VAR))";\
	)

### --- Cleanup targets --- ###

.PHONY: clean
clean: clean-build clean-install clean-test  ## clean everything

.PHONY: clean-build
clean-build:	## clean build caches
	@$(ECHO) "$(_INFO)Removing build caches..."
	@-rm -fr "$(BUILD_DIR)/"
	@-rm -fr "$(APP_ROOT)/CMakeFiles/"
	@-rm -fr "$(APP_ROOT)/CMakeCache.txt"

.PHONY: clean-build-artefacts
clean-build-artefacts:  ## remove CMake build artefacts, but leave configurations and directories intact
	@$(ECHO) "$(_INFO)Removing build artefacts..."
	@-test -f "$(BUILD_DIR)/Makefile" && $(MAKE) -C "$(BUILD_DIR)" clean || \
		$(ECHO) "$(_DEBUG)No Makefile callable in build directory"

.PHONY: clean-install
clean-install:	## clean output install locations
	@$(ECHO) "$(_INFO)Removing install outputs..."
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

.PHONY: clean-test
clean-test:  ## remove test artefacts
	@-rm -f "$(DEMO_LOG_FILE)"

### --- Project targets --- ###

.PHONY: build
build:	## build C++ library extensions from source (refer to CMake variables to configure build)
	@$(ECHO) "$(_INFO)Building C++ libraries..."
	@$(foreach VAR,$(BUILD_CONFIG_VARS),\
		[ ! -z "$($(VAR))" ] || $(ECHO) "$(_WARN)Undefined variable: $(VAR)";\
	)
	@mkdir -p "$(BUILD_DIR)"
	@cd "$(BUILD_DIR)" && \
		$(CMAKE) $(CMAKE_DEBUG) \
			-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
			-DTORCH_DIR=$(TORCH_DIR) \
			-DTORCHVISION_DIR=$(TORCHVISION_DIR) \
			-DOPENCV_DIR=$(OPENCV_DIR) \
			-DCLI11_DIR=$(CLI11_DIR) \
			-DPLOG_DIR=$(PLOG_DIR) \
			-DWITH_PYTHON=OFF \
			-DWITH_TESTS=ON \
			-DWITH_TEST_BENCH=ON \
		"$(APP_ROOT)"
	@cd "$(BUILD_DIR)" && make -j $(shell nproc)

.PHONY: install
install: install-cpp  ## alias to 'install-cpp'

.PHONY: install-cpp
install-cpp: build  ## install built C++ libraries
	@$(ECHO) "$(_INFO)Installing as C++ libraries..."
	@cd "$(BUILD_DIR)" && make install

.PHONY: install-python
install-python:  ## install library extension with Python/C++ bindings into the current Python environment
	@$(ECHO) "$(_INFO)Installing as Python package..."
	python setup.py install

.PHONY: test-bench-help
test-bench-help: clean-test  ## call help command of TestBench application (attempts building it if it doesn't exist)
	@test -f "$(BUILD_DIR)/TestBench/TestBench" || ( \
		$(ECHO) "$(_WARN)TestBench was not found. Attempting to build and install it." && \
		$(MAKE) -j $(shell nproc) build \
	)
	@bash -c '$(BUILD_DIR)/TestBench/TestBench --help'

# add conda
.PHONY: test-bench-demo
test-bench-demo: build clean-test  ## call TestBench application with demo parameters (must have access to data drives)
	@$(ECHO) "$(_INFO)Starting TestBench demo..."
	$(BUILD_DIR)/TestBench/TestBench \
		--arch $(DEMO_MODEL) \
		--optim $(DEMO_OPTIM) \
		--train $(DEMO_DATA_ROOT_DIR)/train \
		--valid $(DEMO_DATA_ROOT_DIR)/val \
		--max-epochs $(DEMO_MAX_EPOCHS) \
		--batch-size $(DEMO_BATCH_SIZE) \
		--workers $(DEMO_WORKERS) \
		--extension $(DEMO_DATA_EXT) \
		--logfile "$(DEMO_LOG_FILE)" \
		--$(DEMO_LOG_LEVEL) \
		$(DEMO_XARGS)

### --- Versioning targets --- ###

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

.PHONY: _bump_install
_bump_install:
	@-bash -c '$(CONDA_CMD) test -f "$(CONDA_ENV_PATH)/bin/bump2version" || pip install $(PIP_XARGS) bump2version'

.PHONY: bump
bump: _bump_install	## bump version using VERSION specified as user input (make VERSION=<X.Y.Z> bump)
	@$(ECHO) "$(_INFO)Updating version..."
	@[ "${VERSION}" ] || ( $(ECHO) "$(_ERROR) 'VERSION' is not set"; exit 1 )
	@-bash -c '$(CONDA_CMD) bump2version $(BUMP_XARGS) --new-version "${VERSION}" patch;'

.PHONY: version
version: _bump_install ## display the current version of the application
	@$(ECHO) -n "$(_INFO)Current version: "
	@-bump2version --allow-dirty --list --dry-run patch | grep current_version | cut -d '=' -f 2
