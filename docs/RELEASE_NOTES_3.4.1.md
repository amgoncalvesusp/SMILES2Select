# SMILES2Select 3.4.1 — complete selection strategies and Tiny

This release fixes incomplete transfer of S2S-Decision choices into the unified application. Version 3.4.0 shipped only three validation-selected baselines; 3.4.1 ships all twelve previously trained packages, including Tiny for all three supported tasks. The original choices and historical model hashes are preserved.

## Selection changes

- **Four model alternatives per task:** logistic regression + Morgan, gradient boosting with scalar inputs, gradient boosting + Morgan, and Tiny. Tasks remain Q72547_WT/IC50, P0DMS8_WT/Ki and Q07869_WT/EC50, each with pActivity ≥ 6 as the active-label threshold. Tiny and the additional baselines are experimental comparators; model complexity does not establish better selection.
- **Explanations before selection:** readable estimator/input labels distinguish validation-selected baselines from comparators. **Model guide and selection tips...** explains each model's inputs, purpose, limitations and how to compare baskets under the same conditions.
- **Restored chemical-only ranking:** **Rank by QED only** provides explicit QED ordering, independent of property weights and Pareto ranking. It estimates chemical desirability, not biological activity.
- **Restored minimum-scaffold control:** **Minimum molecular cores** shares constraints with native and model selection, scenarios, saved sessions and export recipes. Unmet coverage is reported. Existing 3.4.0 sessions remain readable.
- **Preserved decision workflow:** preview leaves the basket unchanged; adoption is explicit and reversible. Model scores, selected IDs and hashes survive session reopen and export. **Create selection** applies the native strategy; **Adopt proposal** applies the model preview.

## Installers and downloads

- **Windows installer:** `SMILES2Select-Setup-3.4.1.exe`.
- **Linux x86_64 installer:** `SMILES2Select-Setup-3.4.1-linux-x86_64.run`. Run `bash SMILES2Select-Setup-3.4.1-linux-x86_64.run` as your normal user. It adds the application, launcher and desktop entry. Default uninstall command: `bash ~/.local/share/smiles2select/uninstall.sh`.
- **Portable bundles:** `SMILES2Select-windows.zip` and `SMILES2Select-linux.tar.gz`.
- **Python wheel**, Windows native-dependency manifest and `SHA256SUMS-v3.4.1.txt` are also attached.

Desktop inference uses ONNX Runtime CPU and does not require PyTorch or a training environment. Training remains an optional advanced operation in a separately configured Python runtime. Source installation requires Python 3.11+ and RDKit 2026.3.5.

## Scientific scope

Every model is restricted to its declared target, endpoint and threshold. Scores do not guarantee activity and are not comparable across tasks. The additional packages were not retrained or selected by their test results for this release. Training references remain restricted to training partitions. Qualification, historical training observations and model-specific evidence are documented in the [catalog qualification](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.4.1/docs/MODEL_CATALOG_QUALIFICATION.md) and distributed model cards. The article benchmark and its frozen results were not changed.

Application code remains MIT-licensed; bundled model resources retain their separate CC BY-SA 4.0 notices.

**Português:** versão 3.4.1 recupera os nove comparadores ausentes, incluindo Tiny, além da seleção QED-only e do mínimo de scaffolds. Cada modelo tem explicação e dicas na interface. Instaladores Windows e Linux acompanham este release. Prévia, adoção explícita, histórico, sessões e exportação continuam no mesmo aplicativo. Consulte o [README](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.4.1/README.md).
