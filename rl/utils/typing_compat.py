from __future__ import annotations

import collections.abc as cabc
import typing


def ensure_typing_extensions_compat() -> None:
    """
    Backfill missing typing_extensions symbols from Python's built-in typing module.

    The local Anaconda environment ships an older typing_extensions, while the
    installed torch/SB3 stack expects newer names such as TypeIs, Self,
    dataclass_transform, and deprecated. Python 3.13 already provides most of
    them in typing, so this patch keeps the repo runnable without forcing an
    environment reinstall.
    """
    try:
        import typing_extensions as te
    except Exception:
        return

    for name in dir(typing):
        if hasattr(te, name):
            continue
        try:
            setattr(te, name, getattr(typing, name))
        except Exception:
            pass

    if not hasattr(te, "Buffer") and hasattr(cabc, "Buffer"):
        te.Buffer = cabc.Buffer

    if not hasattr(te, "deprecated"):
        def deprecated(message, /, *, category=DeprecationWarning, stacklevel=1):
            def decorator(obj):
                try:
                    obj.__deprecated__ = message
                except Exception:
                    pass
                return obj
            return decorator

        te.deprecated = deprecated
