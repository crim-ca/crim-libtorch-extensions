#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import subprocess
from setup import TORCH_EXTENSION_NAME


def python_code_as_command(python_code):
    python_code = inspect.cleandoc(python_code)
    return python_code.replace("TORCH_EXTENSION_NAME", TORCH_EXTENSION_NAME).replace("\n", ";")


def test_import():
    code = """
    import {TORCH_EXTENSION_NAME}
    from {TORCH_EXTENSION_NAME}.efficientnet import EfficientNet
    
    print(EfficientNet.__name__)
    """
    res = subprocess.Popen(
        ["python", "-c", python_code_as_command(code)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    assert res == "EfficientNet"
