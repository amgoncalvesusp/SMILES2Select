"""Registry of molecular descriptors.

Each descriptor carries its provenance (provider, function, toolkit version,
unit, precision) so a report can state exactly how a number was produced and
which published method it is - or is not - compatible with.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from rdkit import Chem

from smiles2select.app_metadata import rdkit_version
from smiles2select.chemistry import preparability as prep
from smiles2select.chemistry import rdkit_descriptors as rd


@dataclass(frozen=True)
class DescriptorDefinition:
    """Immutable description of one computable molecular property."""

    id: str
    label: str
    provider: str
    function: str
    compute: Callable[[Chem.Mol], float | int]
    unit: str | None = None
    precision: int = 4
    dtype: str = "float"
    dependencies: tuple[str, ...] = ()
    compatibility: str = ""

    def version(self) -> str:
        return rdkit_version() if self.provider == "rdkit" else "n/a"

    def as_metadata(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "unit": self.unit,
            "provider": self.provider,
            "function": self.function,
            "version": self.version(),
            "precision": self.precision,
            "dtype": self.dtype,
            "dependencies": list(self.dependencies),
            "compatibility": self.compatibility,
        }


class UnknownDescriptorError(KeyError):
    """Raised when a profile references a descriptor that is not registered."""


class DescriptorRegistry:
    """Mutable collection of :class:`DescriptorDefinition` objects."""

    def __init__(self, definitions: Iterable[DescriptorDefinition] = ()) -> None:
        self._definitions: dict[str, DescriptorDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: DescriptorDefinition) -> None:
        if definition.id in self._definitions:
            raise ValueError(f"descriptor already registered: {definition.id}")
        self._definitions[definition.id] = definition

    def __contains__(self, descriptor_id: object) -> bool:
        return descriptor_id in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)

    def get(self, descriptor_id: str) -> DescriptorDefinition:
        try:
            return self._definitions[descriptor_id]
        except KeyError as exc:
            raise UnknownDescriptorError(
                f"unknown descriptor '{descriptor_id}'; registered: {sorted(self._definitions)}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def metadata(self, descriptor_ids: Iterable[str] | None = None) -> list[dict[str, object]]:
        chosen = tuple(descriptor_ids) if descriptor_ids is not None else self.ids()
        return [self.get(descriptor_id).as_metadata() for descriptor_id in chosen]

    def compute(
        self, mol: Chem.Mol, descriptor_ids: Iterable[str]
    ) -> dict[str, float | int | None]:
        """Compute the requested descriptors for one molecule.

        A descriptor that raises is reported as ``None`` rather than aborting
        the whole record: a single failing property must not discard the other
        twenty. Rules evaluated against a ``None`` value fail closed and are
        reported as unevaluable.
        """
        values: dict[str, float | int | None] = {}
        for descriptor_id in descriptor_ids:
            definition = self.get(descriptor_id)
            try:
                values[descriptor_id] = definition.compute(mol)
            except Exception:
                values[descriptor_id] = None
        return values


def _definition(
    descriptor_id: str,
    label: str,
    function: str,
    compute: Callable[[Chem.Mol], float | int],
    *,
    unit: str | None = None,
    precision: int = 4,
    dtype: str = "float",
    compatibility: str = "",
) -> DescriptorDefinition:
    return DescriptorDefinition(
        id=descriptor_id,
        label=label,
        provider="rdkit",
        function=function,
        compute=compute,
        unit=unit,
        precision=precision,
        dtype=dtype,
        compatibility=compatibility,
    )


BUILTIN_DESCRIPTORS: tuple[DescriptorDefinition, ...] = (
    _definition(
        "mol_wt", "Molecular weight", "Descriptors.MolWt", rd.mol_wt, unit="Da", precision=2
    ),
    _definition(
        "rdkit_wlogp",
        "LogP (Wildman-Crippen)",
        "Crippen.MolLogP",
        rd.rdkit_wlogp,
        precision=3,
        compatibility="WLOGP as used by SwissADME; not XLOGP3, not MLOGP, not ClogP",
    ),
    _definition("mol_mr", "Molar refractivity", "Crippen.MolMR", rd.mol_mr, precision=2),
    _definition(
        "hbd_lipinski",
        "H-bond donors",
        "rdMolDescriptors.CalcNumLipinskiHBD",
        rd.hbd_lipinski,
        dtype="int",
        precision=0,
        compatibility="Lipinski definition (N-H + O-H)",
    ),
    _definition(
        "hba_lipinski",
        "H-bond acceptors",
        "rdMolDescriptors.CalcNumLipinskiHBA",
        rd.hba_lipinski,
        dtype="int",
        precision=0,
        compatibility="Lipinski definition (N + O)",
    ),
    _definition(
        "tpsa",
        "Topological polar surface area",
        "rdMolDescriptors.CalcTPSA",
        rd.tpsa,
        unit="A^2",
        precision=2,
    ),
    _definition(
        "rotatable_bonds",
        "Rotatable bonds",
        "rdMolDescriptors.CalcNumRotatableBonds",
        rd.rotatable_bonds,
        dtype="int",
        precision=0,
    ),
    _definition(
        "heavy_atom_count",
        "Heavy atoms",
        "Mol.GetNumHeavyAtoms",
        rd.heavy_atom_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "explicit_atom_count",
        "Explicit atoms",
        "Mol.GetNumAtoms",
        rd.explicit_atom_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "total_atom_count",
        "Total atoms (with hydrogens)",
        "Chem.AddHs(mol).GetNumAtoms",
        rd.total_atom_count,
        dtype="int",
        precision=0,
        compatibility="Ghose atom count: all atoms including hydrogens",
    ),
    _definition(
        "carbon_count",
        "Carbon atoms",
        "count(atomicNum == 6)",
        rd.carbon_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "heteroatom_count",
        "Heteroatoms",
        "rdMolDescriptors.CalcNumHeteroatoms",
        rd.heteroatom_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "ring_count",
        "Rings",
        "rdMolDescriptors.CalcNumRings",
        rd.ring_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "aromatic_ring_count",
        "Aromatic rings",
        "rdMolDescriptors.CalcNumAromaticRings",
        rd.aromatic_ring_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "formal_charge",
        "Formal charge",
        "Chem.GetFormalCharge",
        rd.formal_charge,
        dtype="int",
        precision=0,
    ),
    _definition(
        "fraction_csp3",
        "Fraction Csp3",
        "rdMolDescriptors.CalcFractionCSP3",
        rd.fraction_csp3,
        precision=3,
    ),
    _definition(
        "qed",
        "QED",
        "QED.qed",
        rd.qed,
        precision=3,
        compatibility="continuous score in [0, 1]; not a pass/fail rule",
    ),
    _definition(
        "nitrogen_oxygen_count",
        "N + O atoms",
        "count(atomicNum in {7, 8})",
        rd.nitrogen_oxygen_count,
        dtype="int",
        precision=0,
        compatibility="atom count used by CNS-oriented profiles",
    ),
    _definition(
        "sa_score",
        "Synthetic accessibility",
        "Contrib/SA_Score/sascorer.calculateScore",
        rd.sa_score,
        precision=2,
        compatibility="Ertl-Schuffenhauer, 1 (easy) to 10 (hard); ranking score, not a rule",
    ),
    _definition(
        "np_score",
        "Natural-product likeness",
        "Contrib/NP_Score/npscorer.scoreMol",
        rd.np_score,
        precision=3,
        compatibility="Ertl-Roggo-Schuffenhauer, about -5 (synthetic) to +5 (natural-like); "
        "ranking score, neither a rule nor a classification",
    ),
    _definition(
        "undefined_stereocenters",
        "Undefined stereocenters",
        "Chem.FindPotentialStereo",
        prep.undefined_stereocenters,
        dtype="int",
        precision=0,
        compatibility="counts unspecified tetrahedral atoms and double bonds",
    ),
    _definition(
        "defined_stereocenters",
        "Defined stereocenters",
        "Chem.FindPotentialStereo",
        prep.defined_stereocenters,
        dtype="int",
        precision=0,
        compatibility="counts specified tetrahedral atoms and double bonds",
    ),
    _definition(
        "fragment_count",
        "Fragment count",
        "Chem.GetMolFrags",
        prep.fragment_count,
        dtype="int",
        precision=0,
        compatibility="number of disconnected fragments",
    ),
    _definition(
        "largest_ring_size",
        "Largest ring size",
        "RingInfo.AtomRings",
        prep.largest_ring_size,
        dtype="int",
        precision=0,
        compatibility="atoms in largest ring; >=12 marks a macrocycle",
    ),
    _definition(
        "amide_bond_count",
        "Amide bond count",
        "rdMolDescriptors.CalcNumAmideBonds",
        prep.amide_bond_count,
        dtype="int",
        precision=0,
        compatibility="amide bonds, proxy for peptide character",
    ),
    _definition(
        "bridgehead_atom_count",
        "Bridgehead atom count",
        "rdMolDescriptors.CalcNumBridgeheadAtoms",
        prep.bridgehead_atom_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "spiro_atom_count",
        "Spiro atom count",
        "rdMolDescriptors.CalcNumSpiroAtoms",
        prep.spiro_atom_count,
        dtype="int",
        precision=0,
    ),
    _definition(
        "tautomer_count",
        "Tautomer count",
        "MolStandardize.TautomerEnumerator",
        prep.tautomer_count,
        dtype="int",
        precision=0,
        compatibility="plausible tautomers capped at 16",
    ),
)


def default_registry() -> DescriptorRegistry:
    """Registry with every built-in descriptor."""
    return DescriptorRegistry(BUILTIN_DESCRIPTORS)


def descriptor_set_fingerprint(
    registry: DescriptorRegistry, descriptor_ids: Iterable[str]
) -> Mapping[str, object]:
    """Provenance block written to the CONFIG sheet and used as a cache key."""
    chosen = sorted(set(descriptor_ids))
    return {
        "descriptors": chosen,
        "rdkit_version": rdkit_version(),
        "methods": {
            descriptor_id: registry.get(descriptor_id).function for descriptor_id in chosen
        },
    }
