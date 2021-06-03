MAKEFILE_NAME := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
APP_ROOT      := $(abspath $(lastword $(MAKEFILE_NAME))/..)
APP_NAME	  := EfficientNet LibTorch (C++/Python)

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

.PHONY: clean
clean: clean-build clean-install  ## clean everything

.PHONY: clean-build
clean-build:	## clean build caches
	@-rm -fr build/

.PHONY: clean-install
clean-install:	## clean output install locations
	@-rm -fr *.egg
	@-rm -fr *.egg-info
	@-rm -fr bdist
	@-rm -fr dist
	@-rm -fr build/bdist*
	@-rm -fr build/dist*
	@-rm -fr build/lib*
	@-rm -fr build/install*
	@-rm -fr install

.PHONY: build
build:	## build C++ library from source
	@mkdir build
	@cd build
	@cmake ..

.PHONY: install
install:  ## install library extension with Python/C++ bindings
	python setup.py install
