from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bitnarrow")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


def load(*args, **kwargs):
    from bitnarrow.models import load as _load

    return _load(*args, **kwargs)


def apply_patch(*args, **kwargs):
    from bitnarrow.models import apply_patch as _apply_patch

    return _apply_patch(*args, **kwargs)


__all__ = ["__version__", "load", "apply_patch"]
