"""Plain-language scientific guidance for the final-library controls."""

from html import escape

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

STRATEGY_HELP = {
    "qed_only": "Rank eligible molecules by QED only, highest first. Use this transparent "
                "drug-likeness baseline to compare with trained models. Property objectives "
                "are ignored; pins and scaffold/cluster quotas still apply. QED is not a "
                "prediction of target activity. Missing or invalid QED stops selection.",
    "balanced": "Start with favorable property trade-offs, then prefer spread in property "
                "values and available rule margins. This does not estimate biological activity.",
    "pareto_first": "Prefer molecules on better Pareto fronts, then nearer the ideal property "
                    "values. A trade-off means improving one property can worsen another.",
    "diversity_first": "Prefer sparsely populated parts of property space. This can favor "
                       "extreme values; it does not measure pairwise structural diversity.",
    "scaffold_coverage": "Choose one eligible molecule per Murcko core first, then fill remaining "
                         "places under quotas. This does not guarantee biological diversity or a "
                         "global coverage optimum under crossed quotas.",
    "manual_assisted": "Keep eligible pinned choices and rank the remaining places by property "
                       "trade-offs and QED. Pins are also preserved by the other strategies.",
}

OBJECTIVE_HELP = {
    "qed": "QED: drug-likeness estimate (0–1). Higher favors its reference property profile; "
           "it is not an activity or safety prediction. QED already combines several properties.",
    "mol_wt": "Molecular weight (g/mol): lower favors smaller molecules. Choose a desired "
              "interval if size should stay within a study-specific window; smaller is not always better.",
    "rdkit_wlogp": "WLOGP: calculated octanol/water logP (lipophilicity). A desired interval "
                   "often expresses a study goal better than an extreme. It is not measured solubility.",
    "tpsa": "TPSA: topological polar surface area (Å²). Use a justified interval when balancing "
            "polarity; this descriptor alone does not establish permeability.",
    "sa_score": "SA score: estimated synthetic accessibility (1–10). Lower suggests easier "
                "synthesis. It does not provide a synthesis route, availability or price.",
    "np_score": "NP score: natural-product likeness. Higher favors natural-product-like "
                "fragments; it does not prove natural origin, activity or safety.",
}

DIRECTION_HELP = (
    "Higher / lower: continuously prefer that direction. Desired value: prefer the smallest "
    "absolute deviation. Desired interval: all values inside tie; outside, nearer is better. "
    "These rank eligible molecules; they do not add a hard chemical filter."
)


def strategy_help_text(strategy, *, large_pool=False, scaffold_ready=True):
    """Describe available behavior without promising a different large-pool algorithm."""
    text = STRATEGY_HELP.get(strategy, strategy)
    if large_pool and strategy != "qed_only":
        text += " Above 2,000 ranked candidates (not final molecules), weighted property percentiles replace exact Pareto ranking."
        if strategy == "scaffold_coverage":
            text += " Core coverage remains active; percentiles rank alternatives within that policy."
        else:
            text += " The property strategies can give the same selection with identical objectives and quotas."
    if strategy == "scaffold_coverage" and not scaffold_ready:
        text += " Missing molecular cores will be computed when you click Create selection."
    return text


def selection_help_html():
    strategies = "".join(
        f"<li><b>{escape(label)}:</b> {escape(STRATEGY_HELP[key])}</li>"
        for key, label in (
            ("qed_only", "Rank by QED only (drug-likeness)"),
            ("balanced", "Balance properties and representation"),
            ("pareto_first", "Prioritize favorable property trade-offs"),
            ("diversity_first", "Spread across property values"),
            ("scaffold_coverage", "Cover more molecular cores"),
            ("manual_assisted", "Complete my pinned choices"),
        )
    )
    properties = "".join(f"<li>{escape(value)}</li>" for value in OBJECTIVE_HELP.values())
    return f"""
    <h2>Choose a final library with a stated scientific purpose</h2>
    <p>First define what your experiment needs: property quality, representation of
    different cores, a property window, or justified fixed molecules. There is no
    universally best setting and none of these criteria predicts docking success.</p>
    <h3>1. Ask for a count, then apply it</h3>
    <p>The original screening set is the starting library. Enter the desired number
    and click <b>Create selection</b>. Wait for completion, then check the applied
    final count and any warnings before exporting. Editing a number alone does not
    replace the library. For 2,000 requested molecules, check that the applied final
    set contains 2,000; do not infer membership from a crowded map.</p>
    <p>The selector fills places among eligible molecules in deterministic order.
    Fixed eligible choices take precedence, and automatic choices must respect
    quotas. Insufficient candidates or quotas can produce a shortfall; more fixed
    choices than requested can produce an excess. Review those warnings.</p>
    <h3>2. Choose a ranking strategy</h3><ul>{strategies}</ul>
    <p><b>Large-library limitation:</b> above 2,000 candidates in the ranking
    universe, a weighted sum of property percentile ranks replaces exact Pareto
    ordering. The threshold refers to candidates, not the requested final count.
    With the same objectives and quotas, the property strategy names can yield the
    same selection. Core coverage still takes one eligible molecule per core first,
    preferring rarer cores; property percentiles rank alternatives within that policy.
    This approximation is not an exact Pareto front or a fingerprint
    diversity optimization. QED-only always uses descending QED, regardless of
    library size. Review the recorded ranking method.</p>
    <p>For exact Pareto ranking, a molecule is dominated if another is at least as
    good in every active objective and better in at least one. Crowding measures
    separation in these property values. Rule margin describes distance from
    screening thresholds, not experimental confidence.</p>
    <h3>3. Set property objectives and directions</h3><ul>{properties}</ul>
    <p>{escape(DIRECTION_HELP)}</p>
    <p>Choose bounds from your research question and evidence, not from the displayed
    defaults. Missing measurements are not invented; missing objective values get
    the least favorable desirability. Strongly correlated objectives can repeat the
    same preference. Selecting QED together with molecular weight also emphasizes
    a property already represented in QED.</p>
    <h3>4. Limit analogues with quotas</h3>
    <p><b>Maximum per molecular core:</b> caps molecules sharing a Murcko scaffold.
    A value of 1 allows one automatically chosen molecule per core. No limit removes
    that cap. The application computes missing cores when needed. Murcko cores
    summarize ring systems and their linkers; acyclic structures can share an empty
    core. Inspect coverage before applying a strict limit.</p>
    <p><b>Maximum per structural cluster:</b> caps molecules in existing structural
    clusters. It requires complete cluster assignments; a map alone cannot supply
    them. Strict quotas may prevent reaching your count. Pinned choices are preserved
    and may exceed a quota, which must be reported.</p>
    <h3>5. Read the map and export the applied set</h3>
    <p>Property PCA projects descriptors into two dimensions; visible clouds are not
    necessarily structural clusters. Structural UMAP is also a projection. Neither
    map layout chooses molecules. Each final marker represents an actual molecule,
    not a centroid; overlapping markers can hide multiple records. Use the final
    count, inspector and complete exported table to verify membership.</p>
    <h3>6. Document and compare</h3>
    <p>Compare scenarios with one justified change at a time. Record objectives,
    directions, bounds, weights, quotas, pins, ranking method, input hashes and
    software versions, and retain the final IDs and selection recipe. Stable input
    IDs resolve ties; changing input order can change IDs. More selected molecules,
    a high QED or broad map coverage does not establish biological activity.</p>
    <h3>Descriptor sources</h3>
    <p><a href="https://www.rdkit.org/docs/source/rdkit.Chem.QED.html">RDKit QED and original publication</a><br>
    <a href="https://rdkit.org/docs/RDKit_Book.html">RDKit descriptor definitions</a><br>
    <a href="https://greglandrum.github.io/rdkit-blog/posts/2023-12-01-using_sascore_and_npscore.html">RDKit SA / NP scores and original publications</a></p>
    """


class SelectionHelpDialog(QDialog):
    """Scrollable, modeless guidance that does not change selection controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("How to choose selection criteria")
        self.resize(700, 600)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(selection_help_html())
        layout.addWidget(self.browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def show_selection_help(window):
    dialog = getattr(window, "selection_help_dialog", None)
    if dialog is None:
        dialog = SelectionHelpDialog(window)
        window.selection_help_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
