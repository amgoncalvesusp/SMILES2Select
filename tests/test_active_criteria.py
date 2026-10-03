"""Applied criteria are independent of still-editable model and strategy controls."""

from dataclasses import replace

from smiles2select.gui.workspace import active_criteria
from smiles2select.gui.workspace.selection_provenance import AppliedSelection
from smiles2select.selection_intelligence.constrained_selection import SelectionOutcome
from smiles2select.selection_intelligence.objectives import ObjectiveSet
from tests import test_workspace
from tests.test_workspace_model_decision import _proposal

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_live_and_applied_criteria_are_distinct(window, qapp):
    window.show()
    window.target_count.setValue(2)
    window.strategy.setCurrentText("qed_only")
    window._auto_select()
    window.target_count.setValue(4)
    window.per_scaffold.setValue(1)
    text = active_criteria.details(window)
    assert "Applied target: 2" in text
    assert "Current requested count: 4" in text
    assert "qed_only" in text
    assert "ignored" in text
    assert "lip_mw_max" in text and "500" in text
    assert "not applied" in window.selection_summary.text()
    assert "Chemical ranking" in window.selection_summary.text()
    assert "Pins:" in window.selection_summary.text()
    assert window.selection_summary.isVisible()
    window.applied_criteria_button.click()
    assert window._criteria_dialog.isVisible()
    window.methods_action.trigger()
    assert window._methods_dialog.isVisible()


def test_model_summary_survives_picker_change_undo_and_native_replacement(window, tmp_path):
    preview = _proposal(window, tmp_path / "preview")
    window.model_panel.accept_preview(preview)
    window.model_panel.adopt()
    assert "Experimental model ranking" in window.selection_summary.text()
    assert "target-1" in window.selection_summary.text()
    window.model_panel.models.addItem("unadopted", {"name": "unadopted", "compatible": False})
    window.model_panel.models.setCurrentIndex(window.model_panel.models.count() - 1)
    assert "target-1" in active_criteria.details(window)
    assert "unadopted" not in active_criteria.details(window)
    window._auto_select()
    assert "Chemical ranking" in window.selection_summary.text()
    window._undo()
    assert "Experimental model ranking" in window.selection_summary.text()


def test_contextual_summary_uses_frozen_context_smarts_and_independent_risk(window):
    provenance = {"context": {"target": "P22303", "species": "Homo sapiens", "endpoint": "IC50",
        "stage": "lead", "assay_context": "biochemical", "model_version": "B15-seed42"},
        "revisited_profiles": ["lipinski"], "policy_settings": {
        "required_smarts": ["c1ccccc1"], "excluded_smarts": ["[Hg]"], "preferred_smarts": [],
        "default_alert_action": "warn", "profiles": [{"profile_id": "lead_like", "action": "inform", "max_violations": 1}],
        "rule_actions": [], "alert_actions": [], "risk_weight": .2, "risk_exclude_at": None,
        "risk_evidence": {"endpoint": "shsy5y_atp_viability_48h", "source": "PubChem"}}}
    applied = AppliedSelection.capture(constraints=window.constraints(), objectives=ObjectiveSet(), pareto=None,
        strategy="contextual_policy", input_hash="test", provenance=provenance)
    window._apply_selection(SelectionOutcome(selected_ids=(1,)), window.constraints(), applied)
    text = active_criteria.details(window)
    for value in ("P22303", "Homo sapiens", "lead", "c1ccccc1", "[Hg]", "shsy5y_atp_viability_48h", "PubChem", "lipinski"):
        assert value in text
    assert "Experimental contextual ranking" in window.selection_summary.text()
    assert "property objectives are ignored" in text
    window._undo()
    assert "Original pipeline" in window.selection_summary.text()


def test_requested_revisit_keeps_explicit_profile_exclusion_visible(window):
    policy = replace(window.result.config.policy, roles={"lipinski": "exclusion"})
    window.result = replace(window.result, config=replace(window.result.config, policy=policy))
    text = "\n".join(active_criteria._screening(window, {"revisited_profiles": ["lipinski"]}))
    assert "explicit exclusion retained" in text
