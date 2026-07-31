"""Molecular standardization.

Every descriptor depends on the exact structure that reaches RDKit, so the
standardization configuration is part of the cache key (see
:meth:`StandardizationConfig.fingerprint`). Changing salt stripping, charge
handling or tautomer canonicalization invalidates previously cached values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from smiles2select.app_metadata import config_hash, rdkit_version


@dataclass(frozen=True)
class StandardizationConfig:
    """Immutable standardization profile."""

    cleanup: bool = True
    remove_salts: bool = True
    neutralize: bool = False
    canonical_tautomer: bool = False
    remove_stereo: bool = False

    def fingerprint(self) -> str:
        """Hash covering both the configuration and the toolkit version."""
        return config_hash({"standardization": asdict(self), "rdkit": rdkit_version()})


@dataclass(frozen=True)
class StandardizationResult:
    mol: Chem.Mol | None
    standardized_smiles: str
    valid: bool
    error: str | None = None


def standardize(mol: Chem.Mol, config: StandardizationConfig) -> StandardizationResult:
    """Apply the configured standardization steps to ``mol``.

    A failure in any step marks the record invalid rather than raising, so the
    pipeline can report it alongside the other rejected rows.
    """
    try:
        work = Chem.Mol(mol)
        if config.cleanup:
            work = rdMolStandardize.Cleanup(work)
        if config.remove_salts:
            work = rdMolStandardize.FragmentParent(work)
        if config.neutralize:
            work = rdMolStandardize.Uncharger().uncharge(work)
        if config.canonical_tautomer:
            work = rdMolStandardize.TautomerEnumerator().Canonicalize(work)
        if config.remove_stereo:
            Chem.RemoveStereochemistry(work)
        if work is None or work.GetNumAtoms() == 0:
            return StandardizationResult(None, "", False, "standardization emptied the molecule")
        Chem.SanitizeMol(work)
        return StandardizationResult(work, Chem.MolToSmiles(work), True, None)
    except Exception as exc:
        return StandardizationResult(None, "", False, f"standardization failed: {exc}")
