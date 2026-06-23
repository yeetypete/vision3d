"""Exceptions raised by vision3d's Rerun logging."""


class LoggingInputError(ValueError):
    """Raised when logging is called with malformed input.

    Signals a caller bug -- a non-scalar metric, mismatched per-box list
    lengths, an unsupported box format -- as opposed to a transient sink or
    transport failure. Subclasses :class:`ValueError`, so existing
    ``except ValueError`` handlers still catch it, but lets
    :class:`RerunLogger` tell a usage bug apart from a visualization hiccup
    and re-raise it even in best-effort (non-``strict``) mode.
    """
