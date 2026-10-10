"""The one hard-coded exclusion. The curated registry cannot remove it.

The id is the CivitAI model at https://civarchive.com/models/2299623.
The hashes are the SHA-256 values of the files published for that model.
"""
from __future__ import annotations

EXCLUDED_MODEL_IDS = frozenset({"2299623"})

EXCLUDED_SHA256 = frozenset({
    "03930710409f6cf19c2cf543d76f522feab9d7197a9a6f98e854561912fabf7e",
    "3b13d8cd7cc3fcfe86725b0ad01ee5acf7f5cefeedd747a557d321520af2e01f",
})
