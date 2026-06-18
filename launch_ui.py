"""
Fix MLflow UI compatibility with Python 3.14
Patches the missing importlib.abc.Traversable import
"""
import importlib.abc
import importlib.resources.abc

# Monkey-patch: restore Traversable into importlib.abc for Python 3.14+
if not hasattr(importlib.abc, 'Traversable'):
    importlib.abc.Traversable = importlib.resources.abc.Traversable

# Now launch MLflow UI
import subprocess, sys
subprocess.run([sys.executable, "-m", "mlflow", "ui", "--host", "127.0.0.1", "--port", "5000"])
