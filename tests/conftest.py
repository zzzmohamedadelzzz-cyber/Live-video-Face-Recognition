"""
Mock deepface at import time so CI does not need TensorFlow/tf-keras installed.

detector.py does:
    from deepface.modules.representation import represent

We inject a fake module at that path so the import resolves to our MagicMock.
Tests configure mock_repr.represent.return_value / side_effect directly.
"""
import sys
from unittest.mock import MagicMock

mock_repr = MagicMock()
mock_repr.represent.return_value = []

# Build the minimal module hierarchy deepface → modules → representation
mock_modules = MagicMock()
mock_modules.representation = mock_repr

mock_deepface = MagicMock()
mock_deepface.modules = mock_modules

sys.modules["deepface"] = mock_deepface
sys.modules["deepface.modules"] = mock_modules
sys.modules["deepface.modules.representation"] = mock_repr
