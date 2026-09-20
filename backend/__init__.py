"""Compatibility package for the in-progress ``backend/src`` source layout.

The application imports the relocated modules as ``backend.*``. Extending the
package path keeps those imports and the existing ``python -m backend...``
entry points working while the source tree remains under ``backend/src``.
"""

from pathlib import Path

_package_root = Path(__file__).resolve().parent
__path__ = [str(_package_root), str(_package_root / "src")]
