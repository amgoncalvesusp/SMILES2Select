# SMILES2Select 3.4.0 — S2S-Decision in one application

SMILES2Select now combines chemical screening and task-specific model prioritization in one installation, workspace and final basket. Chemical selection remains the default. In the Chemical Space Hub, **Prioritize by model** previews a proposal against the current basket; the proposal changes the selection only after explicit adoption. Adoption is one undoable action.

## What changed

- **Model proposals:** score the chemically eligible molecules before the final count is applied. The proposal respects active count, scaffold and cluster quotas, pins, exclusions and reference restrictions. The Hub shows retained, added and removed molecules and blocks stale or incompatible proposals.
- **Three ready-to-use ONNX models:** Q72547_WT/IC50, P0DMS8_WT/Ki and Q07869_WT/EC50. Each was selected using its validation partition and is restricted to its declared target, endpoint and threshold. Tiny remains optional and experimental; no universal activity model is included. Model data and references have a separate CC BY-SA 4.0 notice and model cards.
- **Complete saved sessions:** `.s2s.sqlite` stores processed evidence, candidate structures, basket, applied criteria, undo/redo cursor, model scores and provenance. A saved session can be reopened for historical inspection and export without the original input or model files. A new model preview still requires its compatible package.
- **Traceable exports:** Excel includes `MODEL_SCORES` while the adopted model selection remains active. Final IDs and original SMILES remain tied to the actual basket. Recipes record model and reference hashes; model selection is labeled separately from the underlying quota allocator.
- **One product, optional training:** the existing `s2s-decision` CLI remains available. `s2s-decision-gui` opens the unified application. **Advanced model tools** runs training/evaluation in a separately configured Python runtime; desktop inference uses ONNX Runtime CPU and contains no PyTorch training stack.
- **Package integrity:** imported model packages are checked before use. ONNX inference executes the same bytes that passed SHA-256 verification, so model files cannot redirect inference to unhashed external tensor data.

## Installation and compatibility

Download the Windows installer or portable bundle, or the Linux portable bundle below. Source installations require Python 3.11+; install `smiles2select[inference]` for model proposals or `smiles2select[train]` in a separate training environment. RDKit is pinned to 2026.3.5 to preserve the bundled models' chemistry contract. Existing chemical workflows remain the default. Older supported selection recipes remain readable; the `.s2s.sqlite` file is the complete workspace format.

Release assets: `SMILES2Select-Setup-3.4.0.exe` (Windows installer), `SMILES2Select-windows.zip` and `SMILES2Select-linux.tar.gz` (portable bundles), `SMILES2Select-manifest.json` (Windows native-dependency audit), and the Python wheel. `SHA256SUMS-v3.4.0.txt` lists downloadable file hashes.

Model scores are task-specific estimates. They do **not** guarantee activity, establish biological efficacy, or justify transferring a model to another target or endpoint. The article benchmark was not altered for this release.

## Verification

Local combined suite: 938 passed, 3 skipped; four central integration modules reached 84% branch-aware coverage. An additional 5,001-record regression confirmed scaffold evidence and final export survive session reopen. The Windows and Linux frozen bundles each passed real inference with all three distributed models; Windows native-dependency verification covered 3,724 files. The Windows installer compiled successfully. Installation/update/uninstall on a clean Windows machine has not yet been exercised; see the [candidate validation report](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.4.0/docs/INTEGRATION_CANDIDATE_REPORT.md).

**Português:** seleção química e priorização por modelo agora compartilham aplicativo, sessão e cesta. Três modelos específicos de alvo acompanham o release; Tiny continua experimental. Prévia não altera a cesta; adoção é explícita e reversível. Sessões podem ser reabertas e exportadas sem os arquivos originais. Escores não garantem atividade. Consulte o [README](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.4.0/README.md) e o [guia da interface](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.4.0/docs/GUI_GUIDE.md).
