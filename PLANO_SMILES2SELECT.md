# Plano de correções e evolução — SMILES2Select

**Repositório:** `https://github.com/amgoncalvesusp/SMILES2Select`
**Versão do plano:** 1.0
**Estado de partida verificado:** commit `08ee805`, versão `2.1.0`, 355 testes passando, `ruff check` limpo, testado com RDKit 2026.03.5 e pandas 3.0.2.

---

## 0. Como usar este documento

Você é um agente de programação executando este plano dentro do repositório SMILES2Select.

### 0.1 Regras invioláveis

1. **NUNCA** execute mais de uma tarefa (`T-x.y`) por vez. Termine, verifique, reporte, e só então avance.
2. **NUNCA** altere arquivos fora da lista "Arquivos permitidos" da tarefa em execução.
3. **NUNCA** refatore, renomeie ou "melhore" código que a tarefa não pediu. Se você vir um problema fora do escopo, escreva-o no relatório final e siga em frente.
4. **NUNCA** invente valores químicos, limiares, listas de átomos suportados por programas de docking, ou citações. Se um valor não está neste documento, PARE e pergunte.
5. **NUNCA** apague ou reescreva testes existentes para fazê-los passar. Se um teste existente quebrar, sua alteração está errada.
6. **NUNCA** adicione uma dependência obrigatória nova. Dependências novas são sempre opcionais (`[project.optional-dependencies]`) e precisam de wheel pré-compilado para Windows e Linux.
7. **PARE E PERGUNTE** antes de: apagar arquivos, alterar o esquema do SQLite existente, mudar a assinatura de qualquer função pública já usada por testes, ou alterar `pipeline/runner.py`.

### 0.2 Protocolo de execução

Para cada tarefa, nesta ordem exata:

1. Leia a seção completa da tarefa.
2. Leia os arquivos listados em "Leia antes".
3. Escreva o código.
4. Escreva os testes.
5. Rode a verificação:
   ```
   ruff check .
   ruff format --check .
   python -m pytest -q
   ```
6. Se algo falhar, corrija. Se falhar duas vezes pela mesma razão, PARE e reporte.
7. Emita o relatório:
   ```
   ✅ T-x.y concluída
   Arquivos criados: ...
   Arquivos modificados: ...
   Testes adicionados: N (todos passando)
   Suíte completa: N passed, M skipped
   ruff: limpo
   Observações fora do escopo: ...
   ```

### 0.3 O que este projeto é

Um seletor de moléculas para triagem virtual. Ele lê bibliotecas de SMILES, calcula descritores moleculares, aplica regras de *drug-likeness* e entrega uma seleção para programas de docking. O usuário é professor de bioquímica; a corretude científica e a rastreabilidade importam mais que elegância de código.

---

## 1. Convenções do repositório

Estas convenções já existem no código. Siga-as sem exceção.

### 1.1 Idioma

| Onde | Idioma |
|---|---|
| Docstrings e comentários | **Inglês** |
| Nomes de variáveis, funções, classes | **Inglês** |
| Strings exibidas ao usuário (GUI, CLI, relatórios, mensagens de aviso) | **Português do Brasil** |
| Rótulos de colunas em abas do Excel | **Inglês abreviado** (`MW`, `TPSA`, `RotB`) |

Exemplo real do repositório (`chemical_space/clustering.py`):

```python
def summary_rows(self) -> list[tuple[str, object]]:
    return [
        ("moléculas agrupadas", int(self.labels.notna().sum())),
        ("clusters", self.cluster_count),
    ]
```

Docstring em inglês, string de saída em português. **Nunca misture.**

### 1.2 Estilo de código

- Todo módulo começa com `from __future__ import annotations`.
- Type hints modernos: `int | None`, `list[str]`, `tuple[str, ...]`. Nunca `Optional[int]` ou `List[str]`.
- `ruff` com `line-length = 100` e `select = ["E4", "E7", "E9", "F", "I", "UP", "C4"]`.
- Dataclasses `@dataclass(frozen=True)` para objetos de configuração e resultado.
- Docstrings explicam **por que**, não **o que**. O repositório inteiro segue isso; imite o tom.

### 1.3 Como adicionar um descritor molecular

Este é o padrão exato. Três arquivos, nesta ordem:

**Passo 1** — `src/smiles2select/chemistry/rdkit_descriptors.py`, função pura:

```python
def undefined_stereocenters(mol: Chem.Mol) -> int:
    """Count stereo elements RDKit can perceive but the input never specified."""
    ...
```

**Passo 2** — `src/smiles2select/chemistry/descriptor_registry.py`, entrada em `BUILTIN_DESCRIPTORS`:

```python
_definition(
    "undefined_stereocenters",
    "Undefined stereocenters",
    "Chem.FindPotentialStereo",
    rd.undefined_stereocenters,
    dtype="int",
    precision=0,
    compatibility="counts unspecified tetrahedral atoms and double bonds",
),
```

**Passo 3** — `src/smiles2select/export/excel.py`, entrada em `DESCRIPTOR_EXPORT_COLUMNS` se o descritor deve aparecer nas abas principais.

### 1.4 Armadilhas conhecidas — leia com atenção

**Armadilha 1 — o planejador de descritores.**
`chemistry/descriptor_planner.py` só calcula descritores que algum perfil, escore ou catálogo de alertas exige. Um descritor novo registrado em `BUILTIN_DESCRIPTORS` **não será calculado** a menos que algo o requisite. Veja como `sa_score` faz isso: `pipeline/config.py::RunConfig.score_ids()` acrescenta `"sa_score"` quando `compute_sa=True`, e `SCORE_DEPENDENCIES` em `descriptor_planner.py` mapeia `"sa_score" -> ("sa_score",)`. Replique esse mecanismo.

**Armadilha 2 — o esquema SQLite é enumerado coluna a coluna.**
`storage/sqlite_store.py::SCHEMA` lista explicitamente `mol_wt REAL, rdkit_wlogp REAL, ...` na tabela `molecule_descriptors`. Note que `sa_score` e `np_score` **não** estão lá — o precedente do projeto é que nem todo descritor entra nessa tabela. **Não altere `molecule_descriptors`.** Crie uma tabela nova quando precisar persistir dados novos.

**Armadilha 3 — o hash de configuração.**
`pipeline/config.py::RunConfig.fingerprint()` produz o hash que invalida o cache persistente. Se você adicionar uma opção que muda o resultado de um cálculo, ela **precisa** entrar nesse dicionário, senão o cache devolverá valores errados silenciosamente. Isso é um bug científico, não cosmético.

**Armadilha 4 — descritores rodam em processos separados.**
`pipeline/workers.py` roda em processos `joblib`. As funções precisam ser importáveis no nível do módulo e receber apenas argumentos serializáveis. Objetos caros são construídos uma vez por processo com `@lru_cache`. Siga o padrão de `_alert_engine`.

---

## 2. Decisões já tomadas — não reabra

| Decisão | Razão |
|---|---|
| **NÃO** implementar cálculo de estados de protonação ou pKa | O SMILES2Docking já faz isso de forma sólida na etapa seguinte. Duplicar seria pior: dois métodos divergentes sobre a mesma molécula. |
| **NÃO** enumerar estereoisômeros ou tautômeros como estruturas | Este software **conta e sinaliza** a ambiguidade. A enumeração é responsabilidade da etapa de preparação a jusante. |
| Interface permanece **PySide6 desktop** | Bibliotecas de milhões de moléculas ficam em disco local; uma aplicação web exigiria subir dezenas de GB. |
| **Remover** o suporte a TMAP | `tmap` não tem instalação confiável no Windows. O requisito é estabilidade em Windows e Linux. |
| Clusterização primária **sem dependência nova** | Sphere exclusion usando apenas `DataStructs.BulkTanimotoSimilarity` do RDKit. `hnswlib` fica como acelerador opcional. |
| Motores de docking suportados: **Vina/AD4, GOLD, Glide** | O usuário utiliza os três. Compatibilidade vira arquivo de configuração, nunca código fixo. |

---

## 3. FASE 0 — Correções de baixo risco

**Objetivo:** limpar defeitos triviais e calibrar você com as convenções do repositório antes das tarefas difíceis.
**Duração estimada:** 1 semana.

### T-0.1 — Corrigir o entry point duplicado

**Problema:** `pyproject.toml` declara dois comandos, `smiles2select-gui` e `smiles2select-workspace`, mas `gui/app.py::workspace_main` apenas chama `main()`. São idênticos.

**Arquivos permitidos:** `src/smiles2select/gui/app.py`, `pyproject.toml`.

**O que fazer:** remova a função `workspace_main` de `gui/app.py` e a linha `smiles2select-workspace = ...` de `[project.scripts]` em `pyproject.toml`.

**NÃO faça:** não tente implementar um entry point de workspace independente. O workspace precisa de uma execução concluída para existir.

**Critério de aceite:** `grep -r "workspace_main" src/ tests/` não retorna nada; `python -m pytest -q` passa.

---

### T-0.2 — Sincronizar as dependências do ambiente conda

**Problema:** `pyproject.toml` lista `pyqtgraph>=0.13` e `psutil>=5.9`; `environment/environment.yml` não lista nenhum dos dois.

**Arquivos permitidos:** `environment/environment.yml`.

**O que fazer:** acrescente `pyqtgraph` e `psutil` à lista `dependencies` do YAML, na seção conda (não na de pip).

**Critério de aceite:** os dois nomes aparecem em `environment/environment.yml`.

---

### T-0.3 — Remover os marcadores `ponytail:`

**Problema:** a palavra `ponytail:` aparece como marcador acidental em docstrings, onde deveria ser `Note:`.

**Arquivos permitidos:** `src/smiles2select/chemical_space/clustering.py`, `src/smiles2select/chemistry/fingerprints.py`.

**O que fazer:** substitua `ponytail:` por `Note:` em ambos. Rode `grep -rn "ponytail" src/ tests/` para confirmar que não sobrou nenhum.

**Critério de aceite:** `grep -rn "ponytail" src/ tests/` retorna vazio.

---

### T-0.4 — Mensagem de erro útil em `project_descriptors`

**Problema:** `chemical_space/umap_projection.py::project_descriptors` faz `descriptors[list(features)]` sem verificar se as colunas existem, produzindo um `KeyError` cru. A função equivalente na PCA (`pca_projection.project`) filtra e lança `ValueError` com mensagem.

**Arquivos permitidos:** `src/smiles2select/chemical_space/umap_projection.py`, `tests/test_selection_intelligence_v2.py`.

**O que fazer:** filtre as colunas ausentes exatamente como a PCA faz, e lance `ValueError` com a mesma forma de mensagem quando nenhuma restar.

**Teste a escrever:** `test_umap_descriptors_rejects_missing_columns` — chamar com uma feature inexistente deve levantar `ValueError`, não `KeyError`.

---

### T-0.5 — Remover o TMAP

**Problema:** `tmap` não instala de forma confiável no Windows, e o requisito é estabilidade nas duas plataformas.

**Arquivos permitidos:** `src/smiles2select/chemical_space/tmap_projection.py` (apagar), `src/smiles2select/chemical_space/__init__.py`, `tests/` (remover referências), `README.md`.

**O que fazer:** apague `tmap_projection.py`, remova qualquer importação e qualquer teste que o referencie, remova a menção do README.

**PARE E PERGUNTE** antes de executar esta tarefa: apagar arquivo exige confirmação humana explícita conforme a regra 0.1.7.

**Critério de aceite:** `grep -rin "tmap" src/ tests/ README.md` retorna vazio e a suíte passa.

---

### T-0.6 — Marcar a imputação de descritores ausentes na PCA

**Problema científico:** em `chemical_space/pca_projection.py::standardise`, a linha `filled = usable.fillna(usable.mean())` substitui descritores faltantes pela média antes do z-score. Uma molécula sem TPSA calculada vai parar exatamente no centro daquele eixo, visualmente idêntica a uma molécula genuinamente mediana. Num mapa usado para decidir o que mandar para docking, isso engana o usuário.

**Arquivos permitidos:** `src/smiles2select/chemical_space/pca_projection.py`, `tests/test_selection_intelligence_v2.py`.

**O que fazer:**
1. `standardise` passa a devolver uma terceira saída: `pd.Series` booleana indexada como o resultado, `True` onde ao menos um descritor foi imputado.
2. `Projection` ganha o campo `imputed: pd.Series | None = None`.
3. `Projection.describe()` acrescenta a linha `("moléculas com descritor imputado", int(...))` quando houver alguma.
4. Atualize as chamadas existentes de `standardise` (há uma em `umap_projection.py`).

**NÃO faça:** não mude o comportamento de imputação em si nesta tarefa. Só torne-o visível.

**Testes a escrever:**
- `test_standardise_flags_imputed_rows` — DataFrame com um NaN produz exatamente uma linha marcada.
- `test_projection_describe_reports_imputation`.

---

## 4. FASE 1 — Camada de preparabilidade para docking

**Objetivo:** responder à pergunta que o software hoje não responde — *quanto custa e quão ambíguo é levar esta molécula ao 3D?*
**Duração estimada:** 4 semanas.
**Prioridade:** máxima.

### 4.1 Contexto conceitual

O pipeline atual termina entregando SMILES canônico ao SMILES2Docking. Nada verifica se a molécula é **preparável**. Uma molécula com três estereocentros indefinidos vira oito estruturas 3D distintas sem que ninguém tenha decidido isso. Uma molécula com boro ou selênio pode não ser tipável pelo campo de força do motor escolhido.

Esta camada é a **quinta camada de evidência** do software, ao lado das quatro que o README já descreve:

| Camada | Módulo | Produz |
|---|---|---|
| Regras físico-químicas | `rules/` | aprovação ou reprovação |
| Escores contínuos | `scores/` | ranqueamento |
| Alertas estruturais | `alerts/` | sinalização |
| **Preparabilidade** | **`preparability/`** | **custo e ambiguidade declarados** |
| Política final | `decision/` | seleção |

**Princípio inegociável:** esta camada **nunca reprova uma molécula**. Ela declara custo e ambiguidade. A decisão é do usuário, como todo o resto do software.

### 4.2 Custos medidos — condicionam o desenho

Medições reais em RDKit 2026.03.5, um núcleo:

| Operação | Velocidade |
|---|---|
| `Chem.FindPotentialStereo` | ~24.000 moléculas/s |
| `TautomerEnumerator.Enumerate` (máx. 16) | ~1.000 moléculas/s |

A enumeração de tautômeros é 24× mais cara. Em 10 milhões de moléculas seriam ~2,8 h por núcleo. **Portanto:** estereoquímica entra ligada por padrão; tautômeros entram sob flag explícita.

---

### T-1.1 — Módulo de descritores de preparabilidade

**Arquivos a criar:** `src/smiles2select/chemistry/preparability.py`
**Arquivos permitidos para edição:** nenhum ainda.

**O que fazer:** crie o módulo com as funções abaixo. O código foi verificado com RDKit 2026.03.5 — use exatamente estas APIs.

```python
"""Descriptors that predict the cost of taking a molecule to 3D.

None of these judge a molecule. They state how many distinct structures a
docking run would have to prepare, and where the input left a decision open
that some downstream tool will otherwise make silently.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def undefined_stereocenters(mol: Chem.Mol) -> int:
    """Stereo elements RDKit perceives but the input never specified.

    Covers tetrahedral atoms and double bonds in a single pass. Each one
    doubles the number of structures a docking run must prepare, which is why
    it is counted rather than resolved here.
    """
    elements = Chem.FindPotentialStereo(mol)
    return sum(1 for item in elements if item.specified == Chem.StereoSpecified.Unspecified)


def defined_stereocenters(mol: Chem.Mol) -> int:
    """Stereo elements the input did specify."""
    elements = Chem.FindPotentialStereo(mol)
    return sum(1 for item in elements if item.specified == Chem.StereoSpecified.Specified)


def fragment_count(mol: Chem.Mol) -> int:
    """Disconnected fragments left after standardization.

    More than one means the record is a salt or mixture that survived the
    standardization profile in force.
    """
    return len(Chem.GetMolFrags(mol))


def largest_ring_size(mol: Chem.Mol) -> int:
    """Atoms in the largest ring; 0 for acyclic molecules.

    Twelve or more marks a macrocycle, whose conformational sampling behaves
    differently from an ordinary drug-like ring system.
    """
    rings = mol.GetRingInfo().AtomRings()
    return max((len(ring) for ring in rings), default=0)


def amide_bond_count(mol: Chem.Mol) -> int:
    """Amide bonds, a proxy for peptide-like character."""
    return rdMolDescriptors.CalcNumAmideBonds(mol)


def bridgehead_atom_count(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumBridgeheadAtoms(mol)


def spiro_atom_count(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumSpiroAtoms(mol)
```

**Verificação de comportamento** (valores conferidos, use-os nos testes):

| SMILES | `undefined_stereocenters` | `defined_stereocenters` |
|---|---|---|
| `CC(O)C(N)C=CC1CCCCC1` | 3 | 0 |
| `C[C@H](O)C(=O)O` | 0 | 1 |
| `CCO` | 0 | 0 |
| `O=C(N)/C=C/C` | 0 | 1 |

| SMILES | função | valor |
|---|---|---|
| `CC(=O)O.[Na+]` | `fragment_count` | 2 |
| `C1CCCCCCCCCCC1` | `largest_ring_size` | 12 |
| `CC(=O)NCC(=O)NC` | `amide_bond_count` | 2 |
| `C1CC2CCC1CC2` | `bridgehead_atom_count` | 2 |
| `C1CCC2(CC1)CCCC2` | `spiro_atom_count` | 1 |

**Testes a escrever:** `tests/test_preparability.py`, um teste por função, usando exatamente as tabelas acima.

---

### T-1.2 — Contagem de tautômeros (opcional e cara)

**Arquivos permitidos:** `src/smiles2select/chemistry/preparability.py`, `tests/test_preparability.py`.

**O que fazer:** acrescente ao módulo:

```python
#: Enumerating beyond this is pointless: a molecule with more than sixteen
#: plausible tautomers is ambiguous by any standard, and the exact count adds
#: nothing to that verdict while costing a great deal of time.
MAX_TAUTOMERS = 16


def tautomer_count(mol: Chem.Mol) -> int:
    """Plausible tautomers, capped at ``MAX_TAUTOMERS``.

    Roughly twenty-four times more expensive than every other descriptor here,
    which is why it is never computed unless explicitly requested.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(MAX_TAUTOMERS)
    return len(enumerator.Enumerate(mol))
```

**Valores conferidos para os testes:**

| SMILES | `tautomer_count` |
|---|---|
| `CCO` | 1 |
| `O=C1CCCCC1` | 2 |
| `c1ccccc1O` | 2 |
| `CC(=O)CC(=O)C` | 5 |

**NÃO faça:** não chame `tautomer_count` de nenhum caminho de código padrão. Ela só roda sob flag, ligada na T-1.5.

---

### T-1.3 — Custo estimado de estruturas 3D

**Arquivos permitidos:** `src/smiles2select/chemistry/preparability.py`, `tests/test_preparability.py`.

**O que fazer:** acrescente a função que agrega a ambiguidade num único número interpretável:

```python
#: Above this, the count stops being a plan and becomes a warning.
MAX_REPORTED_STRUCTURES = 1024


def estimated_3d_structures(undefined_stereo: int, tautomers: int = 1) -> int:
    """How many distinct structures a preparation step would have to build.

    Each unspecified stereo element doubles the count; each additional
    plausible tautomer multiplies it. The number is capped because past a
    thousand structures the exact value changes no decision - the molecule is
    simply not worth preparing at that price.
    """
    if undefined_stereo < 0 or tautomers < 1:
        raise ValueError("undefined_stereo must be >= 0 and tautomers >= 1")
    total = (2**undefined_stereo) * tautomers
    return min(total, MAX_REPORTED_STRUCTURES)
```

**Testes a escrever:**
- `test_estimated_structures_doubles_per_stereocenter`: `(0, 1) -> 1`, `(1, 1) -> 2`, `(3, 1) -> 8`.
- `test_estimated_structures_multiplies_by_tautomers`: `(2, 3) -> 12`.
- `test_estimated_structures_is_capped`: `(20, 1) -> 1024`.
- `test_estimated_structures_rejects_invalid_input`: `(-1, 1)` e `(1, 0)` levantam `ValueError`.

---

### T-1.4 — Perfis de compatibilidade por motor de docking

**Arquivos a criar:**
- `src/smiles2select/preparability/__init__.py`
- `src/smiles2select/preparability/engines.py`
- `src/smiles2select/preparability/builtins/vina.json`
- `src/smiles2select/preparability/builtins/gold.json`
- `src/smiles2select/preparability/builtins/glide.json`

**Contexto:** o usuário roda Vina/AD4, GOLD e Glide. Os três aceitam conjuntos diferentes de elementos e reagem diferentemente a macrociclos e peptídeos. **Você não sabe quais são esses conjuntos e não deve descobrir por conta própria.**

**Regra dura:** o único conjunto de elementos que este plano afirma ser universalmente aceito é o conjunto orgânico comum: **H, C, N, O, S, P, F, Cl, Br, I**. Qualquer outro elemento (B, Si, Se, As, metais) é sinalizado como *"requer verificação para o motor X"* — **nunca** como incompatível. Se o usuário quiser afirmar mais que isso, ele edita o JSON e preenche o campo `notes` com a fonte.

Este desenho copia deliberadamente o modelo de perfis do projeto: perfis são arquivos JSON, e alterar um limiar não exige tocar no motor.

**Esquema do JSON:**

```json
{
  "id": "vina",
  "name": "AutoDock Vina / AutoDock4",
  "version": "1.0.0",
  "notes": "Conjunto orgânico comum apenas. Elementos fora da lista exigem verificação manual pelo usuário; nenhuma afirmação sobre suporte foi assumida.",
  "common_elements": ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"],
  "flags": [
    {
      "id": "UNCOMMON_ELEMENT",
      "label": "Elemento fora do conjunto orgânico comum",
      "severity": "verify"
    },
    {
      "id": "MULTIPLE_FRAGMENTS",
      "label": "Mais de um fragmento após padronização",
      "severity": "verify",
      "threshold": 1
    },
    {
      "id": "MACROCYCLE",
      "label": "Macrociclo",
      "severity": "info",
      "threshold": 12
    },
    {
      "id": "PEPTIDE_LIKE",
      "label": "Caráter peptídico",
      "severity": "info",
      "threshold": 4
    },
    {
      "id": "UNDEFINED_STEREO",
      "label": "Estereoquímica indefinida",
      "severity": "decide",
      "threshold": 1
    }
  ]
}
```

Crie `gold.json` e `glide.json` com **o mesmo conteúdo**, mudando apenas `id`, `name` e `notes`. Isso é intencional: os três começam idênticos e conservadores, e o usuário diferencia depois com base nas fontes dele. **NÃO invente diferenças entre eles.**

**Severidades:**

| Severidade | Significado |
|---|---|
| `info` | Registrado no relatório, nenhuma ação necessária |
| `verify` | O usuário deve conferir manualmente antes de enviar ao docking |
| `decide` | Exige uma decisão explícita do usuário (enumerar, escolher, ou excluir) |

Nenhuma severidade exclui moléculas.

**`engines.py`** deve fornecer:

```python
@dataclass(frozen=True)
class EngineFlag:
    id: str
    label: str
    severity: str
    threshold: float | None = None


@dataclass(frozen=True)
class EngineProfile:
    id: str
    name: str
    version: str
    notes: str
    common_elements: frozenset[str]
    flags: tuple[EngineFlag, ...]


def load_engine(engine_id: str, directory: Path | None = None) -> EngineProfile: ...
def available_engines(directory: Path | None = None) -> tuple[str, ...]: ...
```

Siga o padrão de carregamento de `profiles/loader.py`. Acrescente `preparability/builtins/*.json` a `[tool.setuptools.package-data]` no `pyproject.toml`, senão o pacote instalado não terá os arquivos.

**Testes a escrever:** `tests/test_preparability_engines.py` — os três motores carregam; um id inexistente levanta erro claro; o JSON malformado é rejeitado com mensagem útil.

---

### T-1.5 — Integração ao pipeline

**Arquivos permitidos:** `src/smiles2select/chemistry/descriptor_registry.py`, `src/smiles2select/chemistry/descriptor_planner.py`, `src/smiles2select/pipeline/config.py`, `src/smiles2select/cli.py`, `tests/`.

**PARE E PERGUNTE** se qualquer alteração parecer exigir mudança em `pipeline/runner.py`.

**Passo 1 — registrar os descritores.** Acrescente a `BUILTIN_DESCRIPTORS`, seguindo o padrão de 1.3: `undefined_stereocenters`, `defined_stereocenters`, `fragment_count`, `largest_ring_size`, `amide_bond_count`, `bridgehead_atom_count`, `spiro_atom_count`, `tautomer_count`. Todos com `dtype="int"` e `precision=0`.

**Passo 2 — declarar as dependências.** Em `descriptor_planner.py::SCORE_DEPENDENCIES`, acrescente:

```python
"preparability": (
    "undefined_stereocenters",
    "defined_stereocenters",
    "fragment_count",
    "largest_ring_size",
    "amide_bond_count",
),
"preparability_tautomers": ("tautomer_count",),
```

**Passo 3 — opções de configuração.** Em `pipeline/config.py::RunConfig`, acrescente os campos:

```python
compute_preparability: bool = False
compute_tautomers: bool = False
docking_engine: str | None = None
```

Em `score_ids()`, acrescente `"preparability"` quando `compute_preparability` e `"preparability_tautomers"` quando `compute_tautomers`.

**Passo 4 — o hash.** Acrescente `compute_preparability`, `compute_tautomers` e `docking_engine` ao dicionário de `fingerprint()`. **Isto não é opcional** — sem isso o cache devolve resultados errados (Armadilha 3).

**Passo 5 — linhas de resumo.** Em `summary_rows()`, acrescente:

```python
("preparabilidade", "calculada" if self.compute_preparability else "desativada"),
("tautômeros", "enumerados" if self.compute_tautomers else "desativado"),
("motor de docking", self.docking_engine or "-"),
```

**Passo 6 — flags de CLI.** Em `cli.py`, seguindo o padrão de `--sa-score`:

```
--docking-readiness              liga a camada de preparabilidade
--tautomers                      enumera tautômeros (caro: ~1.000 moléculas/s)
--docking-engine {vina,gold,glide}   perfil de compatibilidade (padrão: vina)
```

Ambas as construções de `RunConfig` no `cli.py` (linhas ~248 e ~280) precisam receber os novos argumentos. **Verifique as duas.**

**Testes a escrever:** `tests/test_preparability_pipeline.py`
- `test_readiness_off_by_default` — sem a flag, `undefined_stereocenters` não aparece nas colunas de saída.
- `test_readiness_computes_descriptors` — com a flag, aparece e tem valores corretos.
- `test_tautomers_require_explicit_flag`.
- `test_fingerprint_changes_with_readiness` — o hash muda ao ligar a flag.

---

### T-1.6 — Avaliação das flags e tabela de preparabilidade

**Arquivos a criar:** `src/smiles2select/preparability/evaluation.py`

**O que fazer:** função que recebe o DataFrame de descritores mais um `EngineProfile` e devolve um DataFrame esparso com uma linha por (molécula, flag disparada) — no mesmo espírito das tabelas `rule_failures` e `structural_alerts`, que só registram o que de fato ocorreu.

```python
def evaluate(descriptors: pd.DataFrame, engine: EngineProfile) -> pd.DataFrame:
    """One row per molecule and flag actually raised.

    Sparse on purpose: a library where nothing is flagged produces an empty
    table, not a million rows of "fine".
    """
```

Colunas do resultado: `record_id`, `flag_id`, `severity`, `detail`.

O campo `detail` é uma string em português explicando o caso concreto: `"elemento B fora do conjunto comum"`, `"3 estereocentros indefinidos → 8 estruturas"`, `"anel de 14 átomos"`.

Para `UNCOMMON_ELEMENT`, você precisa dos símbolos dos elementos presentes. Adicione um descritor auxiliar em `preparability.py`:

```python
def uncommon_elements(mol: Chem.Mol, common: frozenset[str]) -> tuple[str, ...]:
    """Element symbols outside the given set, in order of first appearance."""
```

Como essa função precisa de um argumento além da molécula, ela **não** entra no `DescriptorRegistry` (que espera `Callable[[Chem.Mol], float | int]`). Calcule-a em `evaluation.py`, a partir do SMILES canônico já disponível no DataFrame.

**Testes a escrever:** `tests/test_preparability_evaluation.py` — molécula limpa não gera linha; molécula com boro gera `UNCOMMON_ELEMENT`; molécula com 3 estereocentros indefinidos gera `UNDEFINED_STEREO` com `detail` mencionando 8 estruturas.

---

### T-1.7 — Saídas: Excel, SQLite e hand-off

**Arquivos permitidos:** `src/smiles2select/export/excel.py`, `src/smiles2select/storage/sqlite_store.py`, `src/smiles2select/export/docking.py`, `tests/`.

**Excel:** nova aba `05_DOCKING_READINESS` com duas partes:
1. Bloco de resumo (pares chave/valor, em português): total de moléculas selecionadas, total de estruturas 3D estimadas, moléculas com estereoquímica indefinida, moléculas com elemento fora do conjunto comum, moléculas macrocíclicas, motor usado.
2. Tabela detalhada, uma linha por flag disparada.

Acrescente também `undefined_stereocenters` e `estimated_3d_structures` a `DESCRIPTOR_EXPORT_COLUMNS`, com rótulos `UndefStereo` e `Est3D`.

**SQLite:** crie **tabela nova** `preparability_flags (record_id INTEGER, flag_id TEXT, severity TEXT, detail TEXT)`. **NÃO altere `molecule_descriptors`** (Armadilha 2).

**Hand-off:** em `export/docking.py::DockingExportOptions`, acrescente:

```python
#: Molecules with unresolved stereochemistry hand a decision to whatever runs
#: next. Refusing to export them by default forces that decision to be made
#: here, where it is recorded, rather than downstream, where it is not.
block_undefined_stereo: bool = True
```

Quando `True` e houver moléculas bloqueadas, `export()` levanta `ValueError` com mensagem em português listando quantas foram bloqueadas e as três opções do usuário: enumerar a jusante, escolher um isômero, ou passar `block_undefined_stereo=False`.

Acrescente a flag de CLI `--allow-undefined-stereo` que desliga o bloqueio.

**Testes a escrever:** `tests/test_preparability_export.py` — a aba é criada; a tabela SQLite é preenchida; o hand-off bloqueia por padrão e passa com a flag.

---

### T-1.8 — Tela de orçamento de docking

**Arquivos permitidos:** `src/smiles2select/gui/pages/results_page.py`, `src/smiles2select/gui/charts.py`, `tests/test_gui.py`.

**O que fazer:** um painel na tela de resultados, visível só quando a preparabilidade foi calculada, mostrando:

```
1.240 moléculas selecionadas
→ 3.910 estruturas 3D estimadas (motor: gold)

  312 com estereoquímica indefinida     [decidir]
   18 com elemento fora do conjunto comum [verificar]
    7 macrocíclicas                      [informativo]
```

Mais um histograma de `estimated_3d_structures` (escala logarítmica no eixo x), usando o `Canvas` que já existe em `gui/charts.py`.

**NÃO faça:** nenhum cálculo pesado nesta tela. Todos os números já vêm calculados do pipeline.

---

## 5. FASE 2 — Escala e módulos órfãos

**Objetivo:** tornar verdadeira a afirmação "milhões de moléculas" e dar interface aos doze módulos que existem, têm testes, e nenhuma tela alcança.
**Duração estimada:** 8 semanas.

### 5.1 Por que estas duas coisas estão fundidas

O usuário classificou "expor módulos órfãos" acima de "corrigir escalabilidade". Elas não são separáveis: dar interface a sensibilidade, robustez e resgate dentro de uma janela que hoje congela com 8.000 moléculas produz telas que ninguém consegue abrir com uma biblioteca real. Cada módulo ganha interface **e** implementação escalável no mesmo momento.

### 5.2 Os dois modos

Nenhuma tela mostra 10⁷ pontos de forma legível, e nenhuma decisão humana opera nessa escala. A plataforma precisa de dois modos distintos:

| | Modo biblioteca | Modo seleção |
|---|---|---|
| Tamanho | 10⁵ a 10⁷ | 10³ a 5·10⁴ |
| Dados | Em disco (Parquet/DuckDB) | Em memória (pandas) |
| Mapa | Tiles de densidade agregados | Pontos individuais |
| Interação | Zoom, filtro, estatística | Clique, lasso, inspeção |
| Pareto | ε-dominância | Dominância exata |
| Clusters | Sphere exclusion | Butina exato opcional |

O funil entre os dois modos é o produto.

### 5.3 Números de referência medidos

Não confie em intuição sobre desempenho. Estes são os números reais do código atual:

**Clusterização Butina** (`chemical_space/clustering.py`):

| n | tempo | pares na lista Python |
|---|---|---|
| 1.000 | 0,23 s | 500 mil |
| 4.000 | 3,33 s | 8,0 M |
| 8.000 | 18,02 s | 32,0 M |

A lista é uma `list` de floats Python (~32 bytes por elemento, não 8). Em 8.000 moléculas são ~1 GB. Em 50.000 seriam ~1,25 bilhão de pares.

**Pareto** (`selection_intelligence/pareto.py`, 4 objetivos):

| n | tempo | pares de dominância armazenados |
|---|---|---|
| 1.000 | 0,12 s | 55 mil |
| 5.000 | 2,64 s | 1,5 M |
| 20.000 | 46,09 s | 24,8 M |

---

### T-2.1 — Camada de serviço

**Arquivos a criar:** `src/smiles2select/services/__init__.py`, `services/space.py`, `services/selection.py`, `services/readiness.py`.
**Arquivos permitidos:** `src/smiles2select/gui/workspace/workspace_window.py`.

**Problema:** `WorkspaceWindow` chama `cluster()` e `pca_projection.project()` diretamente de dentro de um `QMainWindow`. É por isso que `gui/` tem **0% de cobertura de testes** e nada disso é testável.

**O que fazer:** extraia toda a lógica que não é Qt para funções puras que recebem parâmetros e devolvem DataFrames, sem importar nada de PySide6. A janela passa a apenas ligar sinais e chamar serviços.

**Meta mensurável:** `gui/workspace/` sai de 0% para pelo menos 50% de cobertura, medido por:
```
python -m pytest --cov=smiles2select --cov-report=term-missing
```

**NÃO faça:** não mude o comportamento visível. Esta é uma refatoração pura de movimentação de código.

---

### T-2.2 — Trabalho pesado fora do thread da interface

**Problema crítico:** `WorkspaceWindow._build_candidates` chama `cluster()` de forma síncrona **no construtor**, sobre todas as moléculas avaliáveis. E `refresh()` recalcula `pca_projection.project()` inteira a cada clique, undo, ou adição à cesta.

**Arquivos permitidos:** `src/smiles2select/gui/worker.py`, `gui/workspace/workspace_window.py`, `services/`.

**O que fazer:**
1. Generalize `RunWorker` (que já existe em `gui/worker.py`) para uma fila de trabalho com progresso e cancelamento.
2. Nenhum serviço da T-2.1 pode ser chamado direto da thread da interface.
3. A janela abre imediatamente com um estado "calculando", e preenche conforme os resultados chegam.

**Critério de aceite:** abrir o workspace com 50.000 moléculas mostra a janela em menos de 1 segundo, com indicador de progresso.

---

### T-2.3 — Ligar o cache de projeção que já existe

**Problema:** `chemical_space/projection_cache.py` tem 89% de cobertura de testes e **não é importado por nenhum arquivo em `src/`**. É código morto por falta de fiação.

**Arquivos permitidos:** `src/smiles2select/services/space.py`, `gui/workspace/workspace_window.py`.

**O que fazer:** leia `projection_cache.py`, entenda a interface que ele já oferece, e faça o serviço de espaço químico consultá-lo antes de recalcular qualquer projeção.

**Critério de aceite:** trocar de view e voltar não recalcula a PCA. Adicione um teste que conte chamadas com um mock.

---

### T-2.4 — Clusterização escalável sem dependência nova

**Arquivos a criar:** `src/smiles2select/chemical_space/sphere_exclusion.py`
**Arquivos permitidos:** `chemical_space/clustering.py`, `tests/`.

**Algoritmo (leader / sphere exclusion):**

```
líderes = []
para cada molécula m na ordem de entrada:
    similaridades = BulkTanimotoSimilarity(fp(m), [fp(l) para l em líderes])
    se alguma similaridade >= (1 - cutoff):
        atribua m ao líder de maior similaridade
    senão:
        líderes.append(m); m vira seu próprio cluster
```

Complexidade O(n · k), onde k é o número de líderes. **Nunca materializa o triângulo de distâncias.** Usa apenas `DataStructs.BulkTanimotoSimilarity`, que já está no RDKit — nenhuma dependência nova, nenhuma compilação, comportamento idêntico em Windows e Linux.

**Determinismo:** o resultado depende da ordem de entrada. Isso precisa ser declarado no relatório, como o projeto já faz com `Muegge-RDKit ≠ Muegge-XLOGP3`. Ordene as moléculas por SMILES canônico antes de começar, para que a mesma biblioteca sempre produza o mesmo agrupamento.

**Mantenha** o Butina exato disponível como opção declarada para conjuntos pequenos. O método usado entra no `ClusterResult` e aparece no relatório.

**Testes a escrever:** `tests/test_sphere_exclusion.py`
- Determinismo: duas execuções sobre a mesma entrada dão o mesmo resultado.
- Moléculas idênticas caem no mesmo cluster.
- Moléculas muito distintas caem em clusters distintos.
- 20.000 moléculas terminam em menos de 30 segundos (marcar `@pytest.mark.slow`).

---

### T-2.5 — Pareto com ε-dominância

**Arquivos a criar:** `src/smiles2select/selection_intelligence/epsilon_pareto.py`
**Arquivos permitidos:** `selection_intelligence/pareto.py`, `pareto_ranking.py`, `tests/`.

**Problema duplo, de desempenho e de correção científica.** Com milhões de moléculas, a fronteira de Pareto exata é dominada por diferenças de terceira casa decimal em WLOGP — diferenças sem significado químico. Além disso, `dominance_counts` armazena explicitamente cada par de dominância (24,8 milhões de pares em apenas 20.000 moléculas).

**Solução:** ε-dominância. Cada objetivo recebe um ε correspondente à sua incerteza real. Uma molécula só domina outra se for melhor **por mais que ε**. Discretiza-se o espaço numa grade de células de tamanho ε, mantém-se no máximo uma molécula por célula, e a dominância é computada sobre as células.

**Valores padrão de ε** (ordem de grandeza da incerteza dos métodos; o usuário pode sobrescrever):

| Objetivo | ε padrão |
|---|---|
| `rdkit_wlogp` | 0,5 |
| `tpsa` | 5,0 |
| `mol_wt` | 10,0 |
| `qed` | 0,02 |
| `sa_score` | 0,2 |

**NÃO faça:** não invente valores de ε para descritores que não estão nessa tabela. Se um objetivo não tem ε definido, use ε = 0 (dominância exata) e registre isso no relatório.

**Mantenha** a dominância exata disponível para conjuntos pequenos. O método usado aparece no relatório.

**Testes a escrever:**
- `test_epsilon_dominance_reduces_front_size` — a fronteira com ε > 0 é menor que a exata sobre os mesmos dados.
- `test_epsilon_zero_matches_exact_pareto` — com ε = 0 o resultado é idêntico ao `non_dominated_sort` atual.
- `test_epsilon_pareto_scales` — 200.000 linhas × 4 objetivos em menos de 10 segundos (`@pytest.mark.slow`).

---

### T-2.6 — Desejabilidade suave para faixas-alvo

**Arquivos permitidos:** `src/smiles2select/selection_intelligence/objectives.py`, `tests/`.

**Problema:** em `Objective.desirability`, o caso `TARGET_RANGE` calcula `-(below + above)`, o que dá exatamente 0 para tudo que está dentro da faixa. Como dominância exige "estritamente melhor em ao menos um objetivo", todas as moléculas dentro da faixa ficam mutuamente não-dominadas naquele eixo e a primeira fronteira incha artificialmente.

**Solução:** função de desejabilidade suave no estilo Derringer–Suich. Dentro da faixa, o valor cresce continuamente em direção ao centro; fora dela, decai. Isso mantém a ordenação informativa sem mudar mais nada no motor.

**NÃO faça:** não altere os casos `MAXIMIZE`, `MINIMIZE` ou `TARGET_VALUE`. Não mude a regra de que valores ausentes recebem a pior pontuação.

**Testes a escrever:**
- `test_target_range_discriminates_within_range` — dois valores distintos dentro da faixa recebem desejabilidades distintas.
- `test_target_range_centre_is_best`.
- `test_target_range_outside_is_worse_than_inside`.

---

### T-2.7 — Renderização em escala

**Arquivos permitidos:** `src/smiles2select/gui/workspace/views.py`, `chemical_space/density_tiles.py`, `services/space.py`.

**Dois problemas concretos.**

**Problema A:** `views.set_points` constrói um `QBrush` e um `QPen` por molécula, em laço Python sobre o índice. O docstring do próprio arquivo justifica a escolha do pyqtgraph dizendo que um renderizador vetorial "desenha um elemento por molécula e deixa de ser interativo" — e o código então faz exatamente isso.

**Correção:** passe arrays NumPy para `ScatterPlotItem.setData` e use um número pequeno e fixo de pincéis (um por categoria de cor), nunca um por ponto.

**Problema B:** `density_tiles.should_aggregate` existe, com limiar de 20.000 pontos, e **nunca é chamado**.

**Correção:** implemente uma pirâmide de tiles em múltiplos níveis de zoom, calculada uma vez por projeção e persistida junto com ela. Acima do limiar, desenhe o mapa de densidade com a fração selecionada por tile (`selected_share`, que `tiles()` já calcula). Ao aproximar, consulte o viewport e desenhe os pontos reais.

**Critério de aceite:** 1 milhão de pontos renderizam em menos de 2 segundos, e o zoom permanece responsivo.

---

### T-2.8 — Telas para os módulos órfãos

**Arquivos permitidos:** `src/smiles2select/gui/workspace/panels.py`, `views.py`, `workspace_window.py`, `services/selection.py`.

Cada módulo abaixo já está implementado e testado. Falta uma tela.

| Módulo | Tela | Pergunta que responde |
|---|---|---|
| `sensitivity.py` | Curva de sensibilidade de limiar | Quantas moléculas entram e saem se MW passar de 500 para 520? |
| `robustness.py` | Indicador de robustez por molécula | Esta seleção sobrevive a uma pequena mudança de política? |
| `margins.py` | Coluna de margem na inspeção | Por quanto esta molécula passou ou falhou? |
| `borderline.py` | Lista de casos limítrofes | Quais moléculas estão na fronteira da decisão? |
| `counterfactuals.py` | Painel "e se" | Esta molécula entraria se o limite de TPSA fosse 145? |
| `rescue.py` | Resgate por análogos | Há análogos aprovados de uma molécula reprovada? |
| `clustering.coverage` | Barra de cobertura | Quais quimiotipos a seleção final ignora? |

**Faça uma tarefa por módulo** (T-2.8.1 a T-2.8.7), cada uma com seu ciclo de verificação. Não tente fazer as sete de uma vez.

**Regra transversal:** o `WorkspaceWindow` atual expõe apenas dois objetivos, com direção fixa (o primeiro sempre maximiza, o segundo sempre minimiza). O `ObjectiveSet` suporta até 8 objetivos e 4 direções, incluindo faixas-alvo. Amplie a interface para expor o que o motor já faz.

---

### T-2.9 — Persistência de sessão

**Arquivos permitidos:** `src/smiles2select/storage/selection_store.py`, `services/selection.py`, `gui/workspace/workspace_window.py`.

**Problema:** `selection_store.py` existe, cria a tabela `chemical_space_coordinates`, e o workspace nunca o usa. A sessão não é persistida, apenas exportada. Não é possível fechar o programa e retomar a triagem.

**O que fazer:** salvar a sessão (cesta, decisões, justificativas, coordenadas, clusters) ao fechar, e oferecer retomada ao abrir. Para bibliotecas grandes isso é requisito, não conveniência.

---

## 6. FASE 3 — Publicação

**Duração estimada:** 6 semanas.

### T-3.1 — Comparação de políticas
Rodar duas políticas sobre a mesma biblioteca e mostrar o fluxo entre elas. Os presets já guardam decisões e não arquivos, então a infraestrutura existe. Responde diretamente a "o que eu perco se tornar Ghose obrigatório?".

### T-3.2 — Exportação HTML autocontida
Um arquivo único com mapa, clusters e a justificativa de cada escolha. Resolve o caso "compartilhar com colaborador" sem virar aplicação web.

### T-3.3 — Artigo de software
Dois argumentos sustentam a submissão: a camada de preparabilidade como tipo de evidência ausente das ferramentas atuais de triagem, e a ε-dominância calibrada pela incerteza do descritor como resposta a fronteiras espúrias em bibliotecas grandes. Estudo de caso com a linha de β-lactamases do autor, da biblioteca bruta ao conjunto entregue ao docking, com custo de preparo declarado.

---

## 7. Erros a não cometer

1. **Não** adicione colunas a `molecule_descriptors` no SQLite. Crie tabelas novas.
2. **Não** esqueça de adicionar opções novas ao `RunConfig.fingerprint()`. Cache errado é bug científico.
3. **Não** afirme nada sobre quais elementos Vina, GOLD ou Glide aceitam além do conjunto orgânico comum listado na T-1.4.
4. **Não** transforme a camada de preparabilidade em filtro. Ela declara custo; não reprova.
5. **Não** chame `tautomer_count` em caminho de código padrão. É 24× mais cara que os demais descritores.
6. **Não** introduza dependência que exija compilação na instalação. Windows e Linux, wheels prontos.
7. **Não** faça cálculo pesado na thread da interface.
8. **Não** apague nem enfraqueça testes existentes para fazer os seus passarem.
9. **Não** misture idiomas: docstring em inglês, string de usuário em português.
10. **Não** avance para a tarefa seguinte com a suíte vermelha ou o `ruff` sujo.

---

## Apêndice A — Comandos de verificação

```bash
# Ambiente
conda env create -f environment/environment.yml
conda activate smiles2select
pip install -e ".[dev]"

# Após CADA tarefa
ruff check .
ruff format --check .
python -m pytest -q

# Cobertura (metas da Fase 2)
python -m pytest --cov=smiles2select --cov-report=term-missing

# Testes lentos, apenas quando a tarefa pedir
python -m pytest -m slow
```

## Apêndice B — Estado de partida verificado

Medido em RDKit 2026.03.5, pandas 3.0.2, Python 3.12:

- 355 testes passam, 4 skip, em 11,58 s (excluindo `test_gui.py` e `test_workspace.py`, que exigem PySide6).
- `ruff check .` limpo.
- Cobertura total: 68%.
- `chemical_space/`: 57% a 92%.
- `selection_intelligence/`: 84% a 100%.
- **`gui/`: 0%** — cerca de 1.100 statements, dos quais 682 no workspace.

Se qualquer um desses números divergir na sua execução, **PARE e reporte** antes de escrever código: o repositório mudou desde a redação deste plano.
