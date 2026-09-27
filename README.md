# SMILES2Select

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21920596.svg)](https://doi.org/10.5281/zenodo.21920596)

Author: Adriano Marques Gonçalves — Universidade de Araraquara (UNIARA)

## English

### New in 3.4.0: one application, two selection methods

SMILES2Select now includes S2S-Decision in one installation and workspace.
Chemical screening remains the default. After processing a SMILES library,
open **Chemical Space Hub → Prioritize by model**, choose a target, endpoint
and compatible model, then select **Preview model selection**. The preview
scores the chemically eligible pool under the current count, quotas, pins,
exclusions and reference rules. Compare retained, added and removed molecules
before **Adopt proposal**. Adoption is explicit and undoable; preview or
cancellation never changes the basket.

Three validation-frozen ONNX packages ship with this release:

| Target | Endpoint | Model |
| --- | --- | --- |
| Q72547_WT | IC50 | gradient boosting, scalar descriptors |
| P0DMS8_WT | Ki | logistic regression, scalar descriptors + Morgan fingerprint |
| Q07869_WT | EC50 | logistic regression, scalar descriptors + Morgan fingerprint |

These models are specific to their declared tasks. Their scores do not
guarantee activity, and no universal activity model is included. Tiny remains
an experimental option for separately trained packages. Bundled model data
carry a separate CC BY-SA 4.0 notice; application code remains MIT-licensed.
See [model qualification](docs/MODEL_CATALOG_QUALIFICATION.md) for source,
validation choices, hashes and limits.

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
three ONNX packages and work without a separate training installation.

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

### Novidades da versão 3.4.0

SMILES2Select e S2S-Decision agora formam um único aplicativo. A seleção
química continua como padrão. Após processar a biblioteca, abra **Chemical
Space Hub → Prioritize by model**, escolha alvo, endpoint e modelo compatível,
e use **Preview model selection**. A prévia pontua todas as moléculas
quimicamente elegíveis respeitando quantidade, cotas, moléculas fixadas,
exclusões e regras de referência. Compare moléculas mantidas, adicionadas e
removidas antes de **Adopt proposal**. A adoção altera a cesta em uma ação
reversível; a prévia sozinha não altera a seleção.

Esta versão inclui três modelos ONNX escolhidos por validação para tarefas
específicas: Q72547_WT/IC50, P0DMS8_WT/Ki e Q07869_WT/EC50. Nenhum deles
prediz atividade universal ou garante atividade experimental. Tiny permanece
opcional e experimental. Os dados dos modelos têm aviso CC BY-SA 4.0 separado
da licença MIT do código. Consulte a [qualificação dos modelos](docs/MODEL_CATALOG_QUALIFICATION.md).

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
os três modelos ONNX e não exigem instalação de bibliotecas de treinamento.

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
