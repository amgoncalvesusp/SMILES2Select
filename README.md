# SMILES2Select

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21920596.svg)](https://doi.org/10.5281/zenodo.21920596)

Authors: Adriano Marques Gonçalves — Universidade de Araraquara (UNIARA); Daniel Andres Grajales Ruiz — IQ-UNESP

## English

### New in 3.7.1: shutdown cleanup and the updated Chemical Space Hub

Version 3.7.1 carries the linked candidate table, structured 2D inspector,
read-only molecule comparison and compact method controls described below,
with explicit final Qt/plot cleanup. Ordinary window close/reopen remains
available; selection methods, model packages and scientific limits are preserved.

The 3.7.0 source tag is retained: its Linux CI reported 1,947 passing tests and
18 skips, then failed during Qt process shutdown with a segmentation fault
(exit 139). Builds and publication were skipped, so no 3.7.0 installers were
published. See the [3.7.1 release notes](docs/RELEASE_NOTES_3.7.1.md) for the
follow-up and its verification scope.

### New in 3.7.0: linked candidate inspection and molecule comparison

The **Chemical Space Hub** keeps **Requested N**, the applied method, final count,
difference from the request and selection/export actions visible while its forms
scroll. **Configure** shows **Chemical ranking**, **Model (experimental)**,
**Contextual (experimental)** or **Tolerance (experimental)**. Switching forms
does not apply a method: use **Create selection** or the chosen method's explicit
preview/adopt actions. Current chemical-tolerance controls are under
**Configure → Tolerance (experimental)**. **Refresh map** and **Display...** sit
beside the map.

Choose **Table** to inspect every evaluable Hub candidate, including chemical
screening failures. The read-only table uses stable **Record ID** values, so
sorting and duplicate external molecule IDs do not change the inspected record
or final membership. Table, Chemical space, Pareto and preview-table inspection
share the same structured 2D inspector. A green inspection ring is independent
of the gold final-molecule markers; records absent from a map remain inspectable
in the table without invented coordinates.

Open **Compare molecules...**, inspect a molecule and choose **Keep current
molecule as A**, then inspect another record for B. This read-only window stays
open beside the Hub and refreshes current basket states and applied evidence
after decisions, adoption, undo and redo. Keeping A is a temporary view reference;
it does not select, pin or adopt anything. Pending preview evidence stays in the
main inspector, and comparison calculates no new similarity score.

The existing chemical algorithms, model packages, experimental limits, saved
sessions and export formats are preserved. Optional SMARTS patterns still match
locally; external pattern-design help remains optional. AI selection and training
remain **experimental and need broader independent validation**. See the
[Hub guide](docs/CHEMICAL_SPACE_HUB.md), [GUI guide](docs/GUI_GUIDE.md) and
[3.7.0 release notes](docs/RELEASE_NOTES_3.7.0.md).

### New in 3.6.2: Chemical tolerance (experimental)

The **Chemical tolerance (experimental)** panel learns chemical compatibility
from recorded actives at pActivity >= 6.5. General, target-conditioned and custom
profiles use fixed learned limits, pooled distributions, hierarchical sharing
or molecular neighborhoods. The package shows the actual training counts and
calibration scope. These scores describe compatibility with observed chemistry;
they do not establish activity or safety.

Choose a profile, explicitly select reconsiderable rules or native alert entries,
then preview the result. Already eligible molecules remain eligible. Mandatory
restrictions, manual exclusions and locked SMARTS remain binding; unsupported
cases require review. Final selection uses the existing absolute N, quotas and
QED priority. Adoption is undoable, and ordinary exports and saved sessions retain
the frozen evidence. Changing criteria or context cannot silently reuse a threshold
calibrated for another scope.

Custom fitting uses **Validate data / train custom profile** and the existing
cancelable worker. CSV data should include SMILES, exact pActivity or a measured
concentration with units, endpoint, relation, target/species and source identifiers.
Keep unknown activity unknown. Inspect the validation report before fitting:

```bash
python -m s2s_decision tolerance-prepare --input measurements.csv --output tolerance_data
python -m s2s_decision tolerance-train --input tolerance_data --output tolerance_profile --mode general --recipe A
# Optional: adapt shared fit references while keeping its recipe/transforms. Calibrate the new policy separately.
python -m s2s_decision tolerance-train --input tolerance_data --output adapted_profile --mode general --recipe A --base-profile tolerance_profile
# Optional: supply both independent calibration files together.
python -m s2s_decision tolerance-train --input tolerance_data --output calibrated_profile --mode general --recipe A --positive-cal positive_cal.csv --background-cal background_cal.csv --target-retention 0.95 --max-expansion 0.10
# For a narrower policy, add --reconsider-criteria followed by its exact rule/alert IDs.
python -m s2s_decision tolerance-predict --input library.csv --model calibrated_profile --output tolerance_predictions
```

Prepared data or a CSV can be used for training. Missing support or infeasible
calibration prevents automatic rescue. Statistical profiles use the chemistry
and numerical runtime; they do not require a transformer. Frozen desktop training
uses an external compatible Python runtime. See the offline [Methods guide](src/smiles2select/assets/methods.md)
for algorithms, representations A–D, calibration and benchmark interpretation.
The [chemical-tolerance workflow](docs/CHEMICAL_TOLERANCE.md) documents custom
training, adaptation, package validation and interpretation of decisions.
Windows and Linux 3.6.2 installers include this panel and all three statistical profiles.
Three included recipe A profiles contain 3,829 fitted identities across ten tasks.
Their retention/expansion constraints were infeasible, so automatic rescue is
disabled; descriptive evidence and custom adaptation remain available. See the
[included model card](src/s2s_decision/bundled_tolerance_models/MODEL_CARD.md).
The twelve ONNX choices, including Tiny, and the nine contextual packages remain
available in the same workspace. See the [3.6.2 release notes](docs/RELEASE_NOTES_3.6.2.md).

### New in 3.5.0: Methods and experimental contextual selection

Open the **Methods** tab or menu for an offline English guide to every chemical
strategy, model family, training dataset, calibration step and selection limit.
The workspace shows the criteria that produced the current basket separately
from settings for the next selection. AI selection and training remain
**experimental and need broader independent validation**.

All twelve trained ONNX alternatives, including Tiny, remain available through
**Configure → Model (experimental)**. **Configure → Contextual (experimental)**
also includes trained local models for human CA2/Ki, AChE/IC50 and BACE1/IC50.
Select an included package and load it, inspect the preview, then adopt it
explicitly. Nine packages contain three logistic predictors with three
calibration variants each; they are not nine independent biological models.
Their positive label is an exact measurement at or below 1,000 nM.

Contextual controls expose rule and alert evidence, policy settings, optional
SMARTS patterns and a separate review budget. An optional included risk model
covers **SH-SY5Y ATP viability at 48 hours** only. It does not predict general
safety. Decisions, model hashes and context remain available in saved sessions
and exports. A requested molecule count is a budget, not a fixed percentage;
hard constraints may limit the achievable count.

For optional substructure queries, [SMARTS.plus](https://smarts.plus/) helps with
pattern design and inspection. The [SMARTS guide](docs/CONTEXTUAL_SELECTION.md#optional-smarts)
explains required, excluded and preferred patterns, local RDKit checks and method
references. The website is optional; SMILES2Select matches patterns locally.

The larger contextual feature model did not consistently improve retrospective
recovery over its simpler logistic baseline. Hit, lead and fragment settings
configure policies; no stage-success model was trained. Broad multitask and Von
experiments are documented research work, not extra shipped GUI rankers.
Training new models uses the separately configured Python runtime; the included
models run locally without a training environment.

Windows and Linux installers include the Methods guide and model resources.
See [contextual selection](docs/CONTEXTUAL_SELECTION.md),
[Methods](src/smiles2select/assets/methods.md) and the
[3.5.0 release notes](docs/RELEASE_NOTES_3.5.0.md).

### New in 3.4.1: complete model choices and selection controls

SMILES2Select now includes S2S-Decision in one installation and workspace.
Chemical screening remains the default. After processing a SMILES library,
open **Chemical Space Hub → Configure → Model (experimental)**, choose a target,
endpoint and compatible model, then select **Preview model selection**. The preview
scores the chemically eligible pool under the current count, quotas, pins,
exclusions and reference rules. Compare retained, added and removed molecules
before **Adopt proposal**. Adoption is explicit and undoable; preview or
cancellation never changes the basket.

Twelve trained ONNX packages ship with this release: four alternatives for each
of the three tasks below. Version 3.4.0 included only the three baseline choices;
3.4.1 also includes the nine trained comparators, including Tiny.

| Target | Endpoint | Original validation-selected baseline |
| --- | --- | --- |
| Q72547_WT | IC50 | gradient boosting, scalar descriptors |
| P0DMS8_WT | Ki | logistic regression, scalar descriptors + Morgan fingerprint |
| Q07869_WT | EC50 | logistic regression, scalar descriptors + Morgan fingerprint |

Each task offers these four alternatives:

| Model | When to compare it |
| --- | --- |
| Logistic regression + Morgan | A simple supervised baseline using properties, reference context and fingerprint bits. |
| Gradient boosting, scalar inputs | Nonlinear property/context effects without direct fingerprint-bit inputs. |
| Gradient boosting + Morgan | Tests whether direct structural fingerprints add useful information over scalar inputs. |
| Tiny | Experimental neural model combining properties, reference context and Morgan fingerprints; compare against simpler models under the same quotas. |

Choose **Model guide and selection tips...** for each model's inputs, scope,
limitations and practical comparison advice. Labels distinguish the original
validation-selected baseline from experimental comparators. Keep target,
endpoint, candidate pool and quotas fixed when comparing alternatives.
These models are specific to their declared tasks. Their scores do not
guarantee activity, and no universal activity model is included. Tiny remains
experimental and is not the default. Bundled model data
carry a separate CC BY-SA 4.0 notice; application code remains MIT-licensed.
See [model qualification](docs/MODEL_CATALOG_QUALIFICATION.md) for source,
validation choices, hashes and limits.

The native **Selection strategy** menu also restores **Rank by QED only**;
it ranks chemical desirability, not activity. Advanced criteria now include
**Minimum molecular cores**, alongside maximum-per-core and cluster quotas.
Review warnings when a requested minimum is not reached. **Create selection**
applies the chemical strategy; **Adopt proposal** applies a model preview.

Use **Save session** in the Hub to store a `.s2s.sqlite` file. **Open saved
session** in the main window restores processed results, final basket,
criteria, undo/redo history and model scores, even if input files or the model
have moved. Historical inspection and export work without them; a new model
preview still needs the compatible ONNX package. Excel exports include a
`MODEL_SCORES` sheet while the adopted model selection remains active,
alongside the original SMILES, IDs and model hashes.

**Advanced model tools** provides training and evaluation through a separately
configured Python runtime with `smiles2select[train]`. Desktop bundles include
ONNX Runtime CPU for inference, but omit PyTorch and training converters.
Source installations need `smiles2select[inference]` to use models. The legacy
`s2s-decision` CLI remains available; `s2s-decision-gui` opens the unified app.
See the [GUI guide](docs/GUI_GUIDE.md) for the full workflow.

Version 3.5.0 also provides an opt-in **Contextual policy (experimental)**
workflow with included and imported research JSON models, per-rule and per-alert evidence,
optional SMARTS and endpoint-specific risk predictions, and a separate review budget.
See [contextual selection and review](docs/CONTEXTUAL_SELECTION.md) for its
preview/adoption controls, context restrictions and scientific limits.

### New in 3.3.1

Large-library molecular-core selection now calculates missing scaffolds in the
background and applies the requested count. Unapplied edits block GUI exports,
and completed jobs clear their progress messages. The in-app **How to choose
criteria...** guide explains uses, trade-offs and limitations. Scenario studies
verify algorithm version and selected IDs on replay.

### New in 3.3.0

The Chemical Space Hub preserves the completed run's selection, highlights every
final molecule with an explicit export marker and distinguishes overlapping points
from density summaries. Export the current basket directly to SMILES2Docking as
CSV/XLSX with an accompanying audit manifest. Applied criteria survive control
edits and undo/redo; deterministic quota repair improves fulfillment of the
requested count. See [the GUI guide](docs/GUI_GUIDE.md),
[selection methods](docs/SELECTION_STRATEGIES.md) and
[reproducibility](docs/REPRODUCIBILITY.md).


### New in 3.2

Guided controls, explicit A/B scenario previews, selection stability, scalable
cached-descriptor ranking, and optional local Papyrus evidence. The budget is a
molecule count; no prices or currencies are assumed. See the [GUI guide](docs/GUI_GUIDE.md),
[benchmark protocol](docs/SELECTION_BENCHMARK.md) and [Papyrus integration](docs/PAPYRUS.md)
for usage and measured limits. Large scenario pools use explicitly labeled
weighted percentiles instead of exact quadratic Pareto ranking.

SMILES2Select is a multi-rule drug-likeness and chemical-space selection tool
for large SMILES libraries. Version 3.0 adds the Chemical Space Selection Hub:
explicit candidate, reference and background libraries; exact overlap detection;
fingerprint-based reference similarity and novelty; deterministic diversity and
zone quotas; final/reserve outputs; an English interactive workspace; and
machine-readable selection recipes.

Chemical similarity is computed with Morgan fingerprints and exact Tanimoto
verification. 2D map distances are visualization-only and are never used as a
similarity, novelty or selection threshold.

### Installation

Get version 3.7.1 from [GitHub Releases](https://github.com/amgoncalvesusp/SMILES2Select/releases/tag/v3.7.1).
The release includes Windows and Linux installers, portable bundles, a Python
wheel, a Windows native-dependency manifest and `SHA256SUMS-v3.7.1.txt`.

RDKit is easiest to install with conda:

```bash
conda env create -f environment/environment.yml
conda activate smiles2select
pip install -e .
```

Optional capabilities are available as extras:

```bash
pip install -e ".[maps,fastsearch,parquet]"
```

For model proposals from source, install the inference extra. Training uses a
separate runtime and is never required for chemical screening:

```bash
pip install -e ".[inference]"
# In an external training environment only:
pip install -e ".[train]"
```

Python 3.11 or newer is required. RDKit is pinned to 2026.3.5 to preserve the
chemistry contract of the bundled models.

On Windows, the release provides a portable bundle and a Setup executable.
The Setup package installs the complete PyInstaller bundle, including Qt,
RDKit, NumPy and all bundled DLL/PYD files. The published dependency manifest
records the native import verification performed during the build.
Linux releases provide a portable bundle. Both desktop bundles include the
twelve ONNX packages and work without a separate training installation.

Linux also has a per-user installer. Download the `.run` asset and execute:

```bash
bash SMILES2Select-Setup-3.7.1-linux-x86_64.run
```

It installs the application, command launcher and desktop entry without root.
By default, uninstall with `bash ~/.local/share/smiles2select/uninstall.sh`.
See the [GUI guide](docs/GUI_GUIDE.md) for installation paths and model selection.

### Quick start

The graphical application has seven steps in English:

```bash
smiles2select-gui
```

The CLI can run a reference-aware selection directly:

```bash
smiles2select candidates.csv --smiles-column SMILES --id-column ID \
  --reference ecbd.csv --reference-smiles-column SMILES \
  --reference-id-column ID --exclude-reference-duplicates \
  --selection-strategy reference_aware_diversity \
  --final-count 3000 --reserve-count 500 \
  --excel selection.xlsx --database run.sqlite
```

Background/context libraries are explicit and are never selected:

```bash
smiles2select candidates.csv --smiles-column SMILES \
  --reference known.csv --background public-context.csv \
  --selection-strategy reference_novelty --final-count 500
```

### Selection semantics

The pipeline keeps four kinds of evidence separate:

| Layer | Role |
| --- | --- |
| Rules | pass/fail and failure explanations |
| Scores | continuous ranking information such as QED, SA and NP score |
| Alerts | structural flags with configurable inform/warn/penalize/exclude actions |
| Selection policy | final eligibility, strategy, quotas and reserve allocation |

The default policy makes Lipinski and Veber mandatory, calculates the other
selected profiles as informative outputs, ranks with QED, and warns on PAINS
and Brenk. These are heuristics, not predictions of efficacy, safety or oral
bioavailability.

Available selection strategies include `traditional`, `balanced`,
`diversity_first`, `reference_novelty`, `reference_neighborhood`,
`reference_aware_diversity`, `stratified` and `manual_assisted`.

`--reference-search exact` performs exhaustive reference comparison. The
optional `fast` mode uses HNSW only for neighbour discovery and re-ranks the
discovered pool with exact RDKit Tanimoto. It fails explicitly when `hnswlib`
is not installed rather than silently changing method.

### Reproducible recipes

Save and replay the Hub decision layer with JSON:

```bash
smiles2select library.csv --smiles-column SMILES --profiles lipinski \
  --mandatory lipinski --selection-strategy diversity_first \
  --final-count 100 --reserve-count 25 \
  --save-selection-plan selection.selection.json

smiles2select another-library.csv --smiles-column SMILES \
  --selection-plan selection.selection.json
```

Recipes record schema version, toolkit version, fingerprint settings, seed,
library provenance, zones, projection provenance and final/reserve counts.
Legacy 2.0 recipes remain readable.

### Chemical Space Hub

The workspace provides:

- deterministic property PCA plus optional structural UMAP methods;
- separate reference overlays and layers for candidate/reference/background data;
- progressive point rendering and density tiles for large libraries;
- color controls for selection status, Pareto rank and reference similarity;
- lasso selection, Pareto objectives, scaffold/cluster quotas and a live basket;
- method cards and consequence-aware explanations for strategy changes;
- exact duplicate, nearest-reference, novelty and coverage inspectors;
- final and reserve status, undo/redo, audit history and Excel/recipe export.

The overlay is a visual context layer. Fingerprint similarity remains the
scientific metric shown in the inspector and saved in the database.

The optional dockability envelope is available as a built-in operational
profile. It is a preparation-oriented filter, not a biological prediction.
`natural_product_exploration_preset()` keeps it mandatory while treating
classical drug-likeness profiles as informative by default.

### Outputs

Excel exports include summary, final, reserve, excluded, profile, alert,
reference-overlap, reference-similarity, zone-membership, zone-allocation and
configuration sheets when those data exist. SQLite stores descriptors, profile
results, failures, alerts, decisions, reference provenance, exact overlaps,
similarity pairs, reserves, zones and the run configuration. Parquet is
available for large wide tables.

### Performance and cancellation

Descriptor work is chunked, checkpointed and executed in background workers.
The GUI exposes cooperative cancellation; completed chunks remain in the
checkpoint and can be resumed with the same configuration. Large exact Butina
clustering is guarded before quadratic memory allocation. Reference comparison
does not materialize a candidate-by-reference similarity matrix.

Run the deterministic smoke benchmark with:

```bash
python benchmarks/benchmark_hub.py --molecules 10000 --references 2000 \
  --output benchmark-results.json
```

See the detailed methodology in:

- [Chemical Space Hub](docs/CHEMICAL_SPACE_HUB.md)
- [Reference libraries](docs/REFERENCE_LIBRARIES.md)
- [Visualization](docs/CHEMICAL_SPACE_VISUALIZATION.md)
- [Selection strategies](docs/SELECTION_STRATEGIES.md)
- [Selection zones](docs/SELECTION_ZONES.md)
- [Reference-aware selection](docs/REFERENCE_AWARE_SELECTION.md)
- [Natural-product selection](docs/NATURAL_PRODUCT_SELECTION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Large-library performance](docs/LARGE_LIBRARY_PERFORMANCE.md)
- [Examples](examples/README.md)
- [Changelog](CHANGELOG.md)

### Development

```bash
$env:QT_QPA_PLATFORM = "offscreen"  # PowerShell/CI
$env:PYTHONPATH = "src"
python -m pytest -q
python -m ruff check src tests benchmarks
```

The test suite covers chemistry, profiles, policies, reference libraries,
exact/approximate search contracts, zones, projections, progressive density,
recipes, persistence, GUI interaction, cancellation and CLI integration.

---

## Português

### Novidades da versão 3.4.1

SMILES2Select e S2S-Decision agora formam um único aplicativo. A seleção
química continua como padrão. Após processar a biblioteca, abra **Chemical
Space Hub → Configure → Model (experimental)**, escolha alvo, endpoint e modelo
compatível, e use **Preview model selection**. A prévia pontua todas as moléculas
quimicamente elegíveis respeitando quantidade, cotas, moléculas fixadas,
exclusões e regras de referência. Compare moléculas mantidas, adicionadas e
removidas antes de **Adopt proposal**. A adoção altera a cesta em uma ação
reversível; a prévia sozinha não altera a seleção.

Esta versão inclui 12 modelos ONNX: logística + Morgan, gradient boosting
escalar, gradient boosting + Morgan e Tiny para cada tarefa Q72547_WT/IC50,
P0DMS8_WT/Ki e Q07869_WT/EC50. A versão 3.4.0 distribuía apenas três baselines;
os nove comparadores restantes agora acompanham o aplicativo. Rótulos identificam
os baselines originalmente escolhidos por validação. **Model guide and selection
tips...** explica entradas, limites e quando comparar cada alternativa.
Nenhum modelo prediz atividade universal ou garante atividade experimental.
Tiny permanece opcional e experimental. Os dados dos modelos têm aviso CC BY-SA 4.0 separado
da licença MIT do código. Consulte a [qualificação dos modelos](docs/MODEL_CATALOG_QUALIFICATION.md).

A seleção química recupera **Rank by QED only** e os controles avançados incluem
**Minimum molecular cores**. **Create selection** aplica a estratégia química;
**Adopt proposal** aplica a proposta do modelo. Ambas respeitam as restrições
compartilhadas; verifique avisos e a quantidade final antes de exportar.

Use **Save session** para guardar um arquivo `.s2s.sqlite` e **Open saved
session** para recuperar resultados, cesta, critérios, histórico de desfazer/
refazer e escores. A inspeção e a exportação históricas não exigem os arquivos
de entrada nem o modelo original. Uma nova prévia exige o pacote ONNX
compatível. Enquanto a seleção adotada por modelo permanecer ativa, a
exportação Excel inclui `MODEL_SCORES` e a procedência do modelo, preservando
IDs e SMILES originais.

**Advanced model tools** mantém treinamento e avaliação em um Python externo
com `smiles2select[train]`; os bundles desktop incluem apenas a inferência
ONNX CPU. Consulte o [guia da interface](docs/GUI_GUIDE.md) para o fluxo completo.

O SMILES2Select é uma ferramenta de seleção de drug-likeness e espaço químico
com múltiplas regras para grandes bibliotecas de SMILES. A versão 3.0 adiciona
o Chemical Space Selection Hub: bibliotecas explícitas de candidatos,
referências e contexto; detecção exata de sobreposição; similaridade e novidade
em relação às referências por fingerprints; diversidade determinística e cotas
por zonas; saídas finais/de reserva; um workspace interativo em inglês; e
receitas de seleção legíveis por máquina.

A similaridade química é calculada com fingerprints de Morgan e verificação
exata por Tanimoto. As distâncias dos mapas 2D são somente para visualização e
nunca são usadas como limiar de similaridade, novidade ou seleção.

### Instalação

A forma mais simples de instalar o RDKit é usando conda:

```bash
conda env create -f environment/environment.yml
conda activate smiles2select
pip install -e .
```

Capacidades opcionais estão disponíveis como extras:

```bash
pip install -e ".[maps,fastsearch,parquet]"
```

Para usar modelos em uma instalação por código, instale também
`pip install -e ".[inference]"`. O treinamento exige um ambiente externo com
`pip install -e ".[train]"`. Python 3.11 ou superior é obrigatório; RDKit
2026.3.5 está fixado para preservar o contrato químico dos modelos incluídos.

No Windows, o release fornece um bundle portátil e um executável Setup. O
pacote Setup instala o bundle completo do PyInstaller, incluindo Qt, RDKit,
NumPy e todas as DLL/PYD incluídas. O manifesto de dependências publicado
registra a verificação dos imports nativos realizada durante a compilação.
O release Linux fornece um bundle portátil. Os dois bundles desktop incluem
os 12 modelos ONNX e não exigem instalação de bibliotecas de treinamento.
Linux também tem instalador por usuário: execute
`bash SMILES2Select-Setup-3.7.1-linux-x86_64.run`. Ele cria lançador e entrada no
menu de aplicativos, sem root. A desinstalação padrão usa
`bash ~/.local/share/smiles2select/uninstall.sh`.

### Início rápido

A aplicação gráfica possui sete etapas em inglês:

```bash
smiles2select-gui
```

A CLI pode executar diretamente uma seleção orientada por referências:

```bash
smiles2select candidates.csv --smiles-column SMILES --id-column ID \
  --reference ecbd.csv --reference-smiles-column SMILES \
  --reference-id-column ID --exclude-reference-duplicates \
  --selection-strategy reference_aware_diversity \
  --final-count 3000 --reserve-count 500 \
  --excel selection.xlsx --database run.sqlite
```

Bibliotecas de contexto/background são explícitas e nunca são selecionadas:

```bash
smiles2select candidates.csv --smiles-column SMILES \
  --reference known.csv --background public-context.csv \
  --selection-strategy reference_novelty --final-count 500
```

### Semântica da seleção

O pipeline mantém quatro tipos de evidência separados:

| Camada | Função |
| --- | --- |
| Regras | aprovação/reprovação e explicações das falhas |
| Scores | informação contínua de ranking, como QED, SA e NP score |
| Alertas | sinalizações estruturais com ações configuráveis de informar/avisar/penalizar/excluir |
| Política de seleção | elegibilidade final, estratégia, cotas e alocação de reservas |

A política padrão torna Lipinski e Veber obrigatórios, calcula os demais
perfis selecionados como resultados informativos, faz o ranking com QED e gera
avisos para PAINS e Brenk. São heurísticas, não previsões de eficácia,
segurança ou biodisponibilidade oral.

As estratégias de seleção disponíveis incluem `traditional`, `balanced`,
`diversity_first`, `reference_novelty`, `reference_neighborhood`,
`reference_aware_diversity`, `stratified` e `manual_assisted`.

`--reference-search exact` realiza uma comparação exaustiva com as referências.
O modo opcional `fast` usa HNSW somente para descobrir vizinhos e refaz o
ranking do conjunto encontrado com Tanimoto exato do RDKit. Ele falha
explicitamente quando `hnswlib` não está instalado, em vez de alterar o método
silenciosamente.

### Receitas reprodutíveis

Salve e reproduza a camada de decisão do Hub usando JSON:

```bash
smiles2select library.csv --smiles-column SMILES --profiles lipinski \
  --mandatory lipinski --selection-strategy diversity_first \
  --final-count 100 --reserve-count 25 \
  --save-selection-plan selection.selection.json

smiles2select another-library.csv --smiles-column SMILES \
  --selection-plan selection.selection.json
```

As receitas registram a versão do schema, a versão do toolkit, as configurações
de fingerprint, a seed, a procedência da biblioteca, as zonas, a procedência da
projeção e as contagens finais/de reserva. Receitas legadas 2.0 continuam
legíveis.

### Chemical Space Hub

O workspace fornece:

- PCA determinística de propriedades e métodos estruturais UMAP opcionais;
- overlays e camadas separados para dados de candidatos/referências/contexto;
- renderização progressiva de pontos e tiles de densidade para grandes bibliotecas;
- controles de cor para status de seleção, ranking de Pareto e similaridade com referências;
- seleção por laço, objetivos de Pareto, cotas por scaffold/cluster e cesta ao vivo;
- cards de método e explicações sobre consequências das mudanças de estratégia;
- inspetores de duplicatas exatas, referência mais próxima, novidade e cobertura;
- status final e de reserva, desfazer/refazer, histórico de auditoria e exportação para Excel/receita.

O overlay é uma camada de contexto visual. A similaridade por fingerprint
continua sendo a métrica científica mostrada no inspetor e salva no banco de
dados.

O envelope opcional de dockability está disponível como um perfil operacional
embutido. É um filtro orientado à preparação, não uma previsão biológica.
`natural_product_exploration_preset()` o mantém obrigatório, enquanto trata os
perfis clássicos de drug-likeness como informativos por padrão.

### Saídas

As exportações para Excel incluem planilhas de resumo, final, reserva,
excluídos, perfil, alertas, sobreposição com referências, similaridade com
referências, pertencimento a zonas, alocação de zonas e configuração quando
esses dados existem. O SQLite armazena descritores, resultados de perfis,
falhas, alertas, decisões, procedência das referências, sobreposições exatas,
pares de similaridade, reservas, zonas e a configuração da execução. Parquet
está disponível para tabelas largas grandes.

### Desempenho e cancelamento

O processamento de descritores é dividido em chunks, salvo em checkpoints e
executado em workers de background. A GUI oferece cancelamento cooperativo;
chunks concluídos permanecem no checkpoint e podem ser retomados com a mesma
configuração. O clustering exato de Butina é protegido antes da alocação de
memória quadrática. A comparação com referências não materializa uma matriz de
similaridade candidato-por-referência.

Execute o benchmark determinístico de fumaça com:

```bash
python benchmarks/benchmark_hub.py --molecules 10000 --references 2000 \
  --output benchmark-results.json
```

Consulte a metodologia detalhada em:

- [Chemical Space Hub](docs/CHEMICAL_SPACE_HUB.md)
- [Bibliotecas de referência](docs/REFERENCE_LIBRARIES.md)
- [Visualização](docs/CHEMICAL_SPACE_VISUALIZATION.md)
- [Estratégias de seleção](docs/SELECTION_STRATEGIES.md)
- [Zonas de seleção](docs/SELECTION_ZONES.md)
- [Seleção orientada por referências](docs/REFERENCE_AWARE_SELECTION.md)
- [Seleção de produtos naturais](docs/NATURAL_PRODUCT_SELECTION.md)
- [Reprodutibilidade](docs/REPRODUCIBILITY.md)
- [Desempenho em bibliotecas grandes](docs/LARGE_LIBRARY_PERFORMANCE.md)
- [Exemplos](examples/README.md)
- [Changelog](CHANGELOG.md)

### Desenvolvimento

```bash
$env:QT_QPA_PLATFORM = "offscreen"  # PowerShell/CI
$env:PYTHONPATH = "src"
python -m pytest -q
python -m ruff check src tests benchmarks
```

A suíte de testes cobre química, perfis, políticas, bibliotecas de referência,
contratos de busca exata/aproximada, zonas, projeções, densidade progressiva,
receitas, persistência, interação da GUI, cancelamento e integração da CLI.
