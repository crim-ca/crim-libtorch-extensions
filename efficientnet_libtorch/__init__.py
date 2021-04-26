

# retrieve C++ library bindings
# pytorch must always be imported before extension find all references of dynamic linking
import torch
# make subpackages visible directly and transparently from 'efficientnet_libtorch' or whatever 'egg' name employed as install
from efficientnet_core import *
