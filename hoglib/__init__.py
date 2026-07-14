"""hoglib — shared helpers for the Hog Charts build pipeline.

Imported by the repo-root build_*.py scripts (never run directly). Scripts run
from the repo root, so `import hoglib` resolves without any path setup. Keeping
one copy of season detection, JSON writing, slug/zone/scope logic here kills the
drift risk the pipeline used to carry from copy-pasting these across scripts.
"""
