"""Task-balanced, paired summaries of the predeclared local benchmark."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import file_hash, write_json

METHODS = ("tiny", "logistic", "gradient_boosting", "similarity", "qed")
METRICS = {
    "pr_auc": ("raw", "pr_auc"),
    "precision_at_n": ("raw", "precision_at_n"),
    "roc_auc": ("raw", "roc_auc"),
    "brier": ("raw", "brier"),
    "ece": ("raw", "ece"),
    "selection_precision": ("selection", "precision"),
    "selection_recall": ("selection", "recall"),
    "final_count": ("selection", "final_count"),
    "scaffolds_covered": ("selection", "scaffolds_covered"),
    "shortfall": ("selection", "shortfall"),
}
PAIRED_METRICS = ("pr_auc", "precision_at_n", "selection_precision")


def _read_run(root, task, protocol, seed):
    folder = root / "runs" / task / protocol / str(seed)
    identity = {"task_id": task, "split_method": protocol, "seed": seed}
    if not (folder / "result.json").exists():
        return {**identity, "status": "pending", "reason": "result.json absent"}
    try:
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError("result metadata does not match predeclared run")
        if result.get("status") not in {"completed", "excluded", "failed"}:
            raise ValueError("unrecognized run status")
        if result["status"] == "completed":
            if file_hash(folder / "evaluation.json") != result.get("evaluation_sha256"):
                raise ValueError("evaluation checksum does not match result metadata")
            evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
            if evaluation.get("seed") != seed:
                raise ValueError("evaluation seed does not match predeclared run")
            if any(method not in evaluation.get("methods", {}) for method in METHODS):
                raise ValueError("completed evaluation does not include every baseline")
            if any(
                not isinstance(evaluation["methods"][method].get(section), dict)
                for method in METHODS
                for section in ("raw", "selection")
            ):
                raise ValueError("evaluation metric sections must be objects")
            return {**result, "evaluation": evaluation}
        return result
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {**identity, "status": "failed", "reason": f"invalid run artifact: {error}"}


def _value(evaluation, method, metric):
    section, key = METRICS[metric]
    value = evaluation["methods"][method].get(section, {}).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _task_metrics(runs):
    def average(method, metric):
        values = [_value(run["evaluation"], method, metric) for run in runs]
        return float(np.mean(values)) if all(value is not None for value in values) else None

    return {method: {metric: average(method, metric) for metric in METRICS} for method in METHODS}


def _paired(tasks, baseline, metric):
    supported = {
        task: values["tiny"][metric] - values[baseline][metric]
        for task, values in tasks.items()
        if values["tiny"][metric] is not None and values[baseline][metric] is not None
    }
    values = np.array(list(supported.values()), dtype=float)
    interval = None
    interpretation = "insufficient_tasks"
    if len(values) >= 2:
        # Resample tasks, never the correlated seed runs within each task.
        rng = np.random.default_rng(42)
        boot = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
        interpretation = (
            "positive_descriptive_delta"
            if interval[0] > 0
            else "negative_descriptive_delta"
            if interval[1] < 0
            else "inconclusive"
        )
    return {
        "task_count": len(supported),
        "task_ids": list(supported),
        "mean_delta": float(values.mean()) if len(values) else None,
        "ci95": interval,
        "interpretation": interpretation,
    }


def _protocol_summary(runs, task_ids, seed_count):
    grouped = {task: [run for run in runs if run["task_id"] == task] for task in task_ids}
    complete = {
        task: _task_metrics(group)
        for task, group in grouped.items()
        if len(group) == seed_count and all(run["status"] == "completed" for run in group)
    }
    partial = {
        task: {
            "completed_seeds": [run["seed"] for run in group if run["status"] == "completed"],
            "methods": _task_metrics([run for run in group if run["status"] == "completed"]),
        }
        for task, group in grouped.items()
        if task not in complete and any(run["status"] == "completed" for run in group)
    }
    methods = {}
    for method in METHODS:
        metrics = {}
        for metric in METRICS:
            values = [
                task[method][metric]
                for task in complete.values()
                if task[method][metric] is not None
            ]
            metrics[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "task_count": len(values),
            }
        methods[method] = metrics
    return {
        "supported_tasks": len(complete),
        "complete_task_ids": list(complete),
        "partial_tasks": list(partial),
        "partial_results": partial,
        "task_means": complete,
        "methods": methods,
        "paired": {
            baseline: {metric: _paired(complete, baseline, metric) for metric in PAIRED_METRICS}
            for baseline in METHODS
            if baseline != "tiny"
        },
    }


def summarize_experiment(root: str | Path) -> dict:
    """Average seeds within tasks, then tasks equally; keep incomplete runs visible."""
    root = Path(root)
    prepared = json.loads((root / "prepared.json").read_text(encoding="utf-8"))
    if file_hash(root / "plan.json") != prepared.get("plan_sha256"):
        raise ValueError("plan checksum does not match preparation manifest")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    tasks = [task["id"] for task in plan["tasks"]]
    seeds, protocols = plan["seeds"], plan["split_methods"]
    for values in (tasks, seeds, protocols):
        if not values or len(values) != len(set(values)):
            raise ValueError("plan tasks, seeds and protocols must be nonempty and unique")
    if any(
        not isinstance(task, str) or Path(task).name != task or task in {".", ".."}
        for task in tasks
    ):
        raise ValueError("task IDs must be plain directory names")
    if any(protocol not in {"scaffold", "temporal"} for protocol in protocols):
        raise ValueError("unknown split protocol")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    runs = [
        _read_run(root, task, protocol, seed)
        for task in tasks
        for protocol in protocols
        for seed in seeds
    ]
    return {
        "schema_version": 1,
        "plan": plan,
        "expected_runs": len(runs),
        "planned_tasks": plan["tasks"],
        "seeds": seeds,
        "status_counts": dict(Counter(run["status"] for run in runs)),
        "aggregation": "seed mean within task, then equal-weight task mean; complete planned seeds only",
        "bootstrap": {
            "unit": "task",
            "replicates": 2000,
            "seed": 42,
            "confidence": 0.95,
            "interpretation": "descriptive, conditional on selected tasks; no multiple-testing correction",
        },
        "protocols": {
            protocol: _protocol_summary(
                [run for run in runs if run["split_method"] == protocol], tasks, len(seeds)
            )
            for protocol in protocols
        },
        "runs": runs,
    }


def _number(value):
    return "—" if value is None else f"{value:.4f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _markdown(summary):
    lines = [
        "# Benchmark ampliado S2S-Decision",
        "",
        "Resultados retrospectivos do conjunto declarado em plan.json. Nenhuma extrapolação prospectiva.",
        "",
        "Seeds previstas: "
        + ", ".join(map(str, summary["seeds"]))
        + ". Dados e critérios de inclusão: plan.json, reproduzido em summary.json.",
        "",
        f"Execuções previstas: {summary['expected_runs']}. Estados: "
        + ", ".join(f"{key}={count}" for key, count in summary["status_counts"].items())
        + ".",
        "",
        "Média das seeds por tarefa; depois média com peso igual entre tarefas. Agregado principal inclui somente tarefas com todas as seeds previstas concluídas. Métricas ausentes excluem a tarefa somente daquela métrica; denominador informado por célula.",
        "",
    ]
    for protocol, result in summary["protocols"].items():
        lines += [
            f"## {protocol}",
            "",
            f"Tarefas completas: {result['supported_tasks']}. Parciais: {len(result['partial_tasks'])}.",
            "",
            "| Método | AP (tarefas) | Precisão@N (tarefas) | Precisão após seleção (tarefas) | Scaffolds selecionados (tarefas) |",
            "|---|---:|---:|---:|---:|",
        ]
        for method, metrics in result["methods"].items():
            values = [
                f"{_number(metrics[key]['mean'])} ({metrics[key]['task_count']})"
                for key in ("pr_auc", "precision_at_n", "selection_precision", "scaffolds_covered")
            ]
            lines.append(f"| {method} | " + " | ".join(values) + " |")
        lines += [
            "",
            "Diferenças pareadas Tiny − referência. IC 95% descritivo: 2.000 reamostragens de tarefas, seed 42. Uma tarefa isolada não permite IC entre tarefas.",
            "",
            "| Referência | Métrica | Tarefas comuns | Diferença | IC 95% | Interpretação |",
            "|---|---|---:|---:|---|---|",
        ]
        for baseline, metrics in result["paired"].items():
            for metric, values in metrics.items():
                interval = (
                    "—" if values["ci95"] is None else " a ".join(map(_number, values["ci95"]))
                )
                lines.append(
                    f"| {baseline} | {metric} | {values['task_count']} | {_number(values['mean_delta'])} | {interval} | {values['interpretation']} |"
                )
        if result["partial_tasks"]:
            lines += [
                "",
                "Tarefas parciais fora do agregado principal: "
                + ", ".join(map(_cell, result["partial_tasks"]))
                + ". Médias parciais disponíveis em summary.json, sem escolha da melhor seed.",
            ]
        lines += [""]
    lines += [
        "## Limites",
        "",
        "IC contendo zero: resultado inconclusivo. IC acima de zero descreve diferença nestas tarefas; não demonstra superioridade geral. Tarefas podem compartilhar alvos/moléculas; bootstrap não elimina essa dependência. Comparações múltiplas sem correção. Partições temporais continuam retrospectivas; datas agregadas não equivalem à disponibilidade prospectiva. Resultados cobrem somente dados, alvos, endpoints, critérios e protocolo declarados. Seleção do SMILES2Select não é rótulo experimental nem P(advance).",
        "",
        "## Todas as execuções previstas",
        "",
        "| Tarefa | Partição | Seed | Estado | Pool comum | N solicitado | Máximo/scaffold | Segundos | Motivo |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        evaluation = run.get("evaluation", {})
        lines.append(
            f"| {_cell(run['task_id'])} | {run['split_method']} | {run['seed']} | {run['status']} | {evaluation.get('pool', {}).get('common_count', '—')} | {evaluation.get('n_requested', '—')} | {evaluation.get('max_per_scaffold', '—')} | {_number(run.get('elapsed_seconds'))} | {_cell(run.get('reason') or '')} |"
        )
    return "\n".join(lines) + "\n"


def write_report(root: str | Path) -> dict[str, Path]:
    """Save auditable JSON and human-readable Portuguese report."""
    root = Path(root)
    summary = summarize_experiment(root)
    paths = {"summary": root / "summary.json", "report": root / "REPORT.md"}
    write_json(paths["summary"], summary)
    paths["report"].write_text(_markdown(summary), encoding="utf-8")
    return paths
