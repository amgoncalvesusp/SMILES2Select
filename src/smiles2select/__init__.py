"""SMILES2Select — multi-rule drug-likeness selector.

The package separates four distinct kinds of evidence, which must never be
collapsed into a single boolean:

1. Physicochemical rules (:mod:`smiles2select.rules`) -> pass / fail.
2. Continuous scores (:mod:`smiles2select.scores`) -> ranking only.
3. Structural alerts (:mod:`smiles2select.alerts`) -> flagging only.
4. Final selection policy (:mod:`smiles2select.decision`) -> combines the above.

Drug-likeness profiles are heuristics derived from historical drug sets. They
do not predict oral bioavailability, efficacy or safety.
"""

from smiles2select.app_metadata import (
    AFFILIATION,
    APP_NAME,
    APP_VERSION,
    AUTHOR,
    AUTHORSHIP,
    rdkit_version,
)

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "AUTHOR",
    "AFFILIATION",
    "AUTHORSHIP",
    "rdkit_version",
]
