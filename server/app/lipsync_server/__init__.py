"""Local lip-sync inference server (runs alongside Deckard on the same GPU pod).

This module intentionally keeps imports light so the server can start even if
optional heavy ML dependencies are not installed yet.
"""

