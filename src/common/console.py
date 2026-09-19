"""
Console utilities for cross-platform compatibility.

Provides utilities for ensuring consistent console behavior across different
platforms, particularly Windows UTF-8 encoding configuration.
"""

import sys


def ensure_utf8_stdout() -> None:
    """
    Ensure stdout is configured for UTF-8 encoding on Windows.

    This function reconfigures sys.stdout to use UTF-8 encoding on Windows
    systems to handle Unicode characters properly. On other platforms, it
    does nothing as UTF-8 is typically the default.

    Usage:
        from src.common.console import ensure_utf8_stdout
        ensure_utf8_stdout()
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ensure_utf8_stderr() -> None:
    """
    Ensure stderr is configured for UTF-8 encoding on Windows.

    Similar to ensure_utf8_stdout but for stderr. This is useful for
    error messages and debugging output that may contain Unicode characters.

    Usage:
        from src.common.console import ensure_utf8_stderr
        ensure_utf8_stderr()
    """
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def ensure_utf8_streams() -> None:
    """
    Ensure both stdout and stderr are configured for UTF-8 encoding on Windows.

    Convenience function that calls both ensure_utf8_stdout and ensure_utf8_stderr.

    Usage:
        from src.common.console import ensure_utf8_streams
        ensure_utf8_streams()
    """
    ensure_utf8_stdout()
    ensure_utf8_stderr()
