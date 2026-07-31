# SMILES2Select

Seletor multirregra de drug-likeness para grandes bibliotecas de SMILES.

O aplicativo importa bibliotecas, calcula cada propriedade molecular **uma única
vez por estrutura canônica** e avalia as moléculas segundo perfis independentes,
mantendo separados quatro tipos de evidência que costumam ser confundidos:

| Camada | Módulo | O que produz |
| ------ | ------ | ------------ |
| Regras físico-químicas | `rules/` | aprovação ou reprovação |
| Escores contínuos | `scores/` | ranqueamento |
| Alertas estruturais | `alerts/` | sinalização |
| Política final | `decision/` | seleção |

> As regras de drug-likeness são heurísticas derivadas de conjuntos históricos de
> fármacos. Elas **não** predizem biodisponibilidade oral, eficácia ou segurança.
> Uma molécula aprovada em todos os perfis não é, por isso, um fármaco; uma
> molécula reprovada não está, por isso, descartada.

---

## Instalação

RDKit é mais simples de instalar por conda:

```bash
conda env create -f environment/environment.yml
```

```bash
conda activate smiles2select
```

```bash
pip install -e .
```

## Uso

Interface gráfica (PySide6, sete etapas):

```bash
smiles2select-gui
```

Linha de comando:

```bash
smiles2select library.xlsx --smiles-column SMILES --id-column ID --profiles lipinski,veber,ghose,egan,muegge --mandatory lipinski,veber --excel resultados.xlsx --database run.sqlite
```

Listar os perfis disponíveis e suas ressalvas:

```bash
smiles2select --list-profiles
```

Consenso de N perfis em vez de interseção:

```bash
smiles2select library.csv --smiles-column SMILES --mandatory "" --consensus 3:lipinski,veber,ghose,egan,muegge
```

Expressão personalizada:

```bash
smiles2select library.csv --smiles-column SMILES --mandatory "" --expression "(lipinski AND veber) AND qed >= 0.50 AND NOT brenk"
```

Cache persistente entre execuções e saída Parquet:

```bash
smiles2select library.csv --smiles-column SMILES --cache descritores.sqlite --parquet resultados.parquet
```

Salvar a configuração como preset e reutilizá-la em outra biblioteca:

```bash
smiles2select lote1.csv --smiles-column SMILES --mandatory lipinski,veber --save-preset politica_do_grupo.json
```

```bash
smiles2select lote2.csv --smiles-column SMILES --preset politica_do_grupo.json --excel lote2.xlsx
```

Perfis personalizados em diretório próprio:

```bash
smiles2select library.csv --smiles-column SMILES --profile-dir ./meus_perfis --profiles lipinski,meu_perfil
```

---

## Perfis incluídos

| id | Perfil | Espaço químico | Política padrão |
| -- | ------ | -------------- | --------------- |
| `lipinski` | Lipinski (Regra dos Cinco) | oral drug-like | até 1 violação |
| `veber` | Veber | oral drug-like | todos os critérios |
| `ghose` | Ghose | caracterização de espaço | todos os critérios |
| `egan` | Egan — compatível com SwissADME | oral drug-like | todos os critérios |
| `muegge` | Muegge — adaptação RDKit | oral drug-like | todos os critérios |
| `ro3_core` | Rule of Three — Core | fragmentos | todos os critérios |
| `ro3_extended` | Rule of Three — Extended | fragmentos | todos os critérios |
| `lead_like` | Lead-like | leads | todos os critérios |
| `cns_like` | CNS-like — heurística físico-química | SNC | até 1 violação |
| `beyond_ro5` | Beyond Rule of Five — espaço estendido | bRo5 | todos os critérios |

### Ressalvas de implementação declaradas no relatório

- **Egan** aplica limites fixos de WLOGP e TPSA como o SwissADME. O trabalho
  original usou PSA e AlogP98 em um modelo de absorção intestinal passiva; os
  vereditos dependem da definição dos descritores e diferem do modelo publicado.
- **Muegge** aqui usa `Crippen.MolLogP` (RDKit), não XLOGP3:
  `Muegge-RDKit ≠ Muegge-XLOGP3`.
- **Ghose** conta **todos os átomos, incluindo hidrogênios** (`Chem.AddHs`), que
  é a convenção do filtro original — não o número de átomos pesados.
- **Rule of Three** e **lead-like** não são filtros de drug-likeness. Servem,
  respectivamente, para bibliotecas de fragmentos e para pontos de partida de
  otimização.
- **CNS-like** NÃO é o CNS MPO de Wager et al. Aquele modelo exige cLogD(7,4) e
  o pKa do centro mais básico — nenhum dos dois é calculado pelo RDKit — e
  devolve um escore contínuo de desejabilidade, não um veredito. Este perfil
  aplica como cortes explícitos as tendências físico-químicas usualmente
  relatadas para compostos que penetram no SNC. É um filtro grosseiro de espaço
  químico, não uma predição de penetração cerebral.
- **Beyond Rule of Five** descreve onde compostos grandes oralmente absorvidos
  têm sido encontrados; não prediz absorção, que nesse espaço depende de
  comportamento conformacional que nenhum descritor aqui captura. Serve para
  manter macrociclos e peptidomiméticos no relatório em vez de descartá-los
  como falhas de Lipinski.
- Não existe campo genérico `logp`. Cada método tem sua própria coluna
  (`rdkit_wlogp`, e futuramente `xlogp3`, `mlogp`, `user_imported_logp`), para
  que valores calculados por métodos diferentes nunca sejam comparados como
  equivalentes.

## QED e alertas

- **QED** é escore contínuo de 0 a 1. Modos: calcular, ranquear, percentil
  superior, limite definido pelo usuário. Não há limite universal por padrão.
- **PAINS** e **Brenk** (`FilterCatalog` do RDKit) geram **ALERTA**, não
  reprovação. Cada catálogo aceita ação configurável: informar, advertir,
  penalizar ou excluir. O padrão advertir não remove nenhuma molécula.
- SMARTS personalizados são compilados e validados **antes** da execução.

## Escore de acessibilidade sintética

`--sa-score` calcula o SA score de Ertl-Schuffenhauer (1 = fácil de sintetizar,
10 = difícil), do módulo contrib do RDKit. Como o QED, é escore contínuo para
ranqueamento e relatório — nunca regra de aprovação. Um valor alto significa
que as contribuições dos fragmentos são incomuns frente ao conjunto de
referência, não que a molécula seja insintetizável.

## Diversidade e scaffolds

Diversidade depende do conjunto, não da molécula: se um composto entra entre os
escolhidos depende do que já foi escolhido. Por isso ela roda **depois** de
todas as regras, sobre as moléculas já selecionadas, e só pode reduzir — nunca
resgatar uma molécula excluída. Cada remoção fica registrada com o seu motivo,
como qualquer outra exclusão.

```bash
smiles2select library.csv --smiles-column SMILES --diverse 500 --docking para_docking.xlsx
```

```bash
smiles2select library.csv --smiles-column SMILES --per-scaffold 1
```

- `--diverse N` — MaxMin sobre fingerprints Morgan (r=2, 2048 bits), semente
  fixa para reprodutibilidade. O relatório mostra a similaridade média e a
  contagem de scaffolds antes e depois.
- `--per-scaffold N` — no máximo N moléculas por scaffold de Murcko; barato,
  sem fingerprints, quando o objetivo é apenas impedir que um quimiotipo domine.

Medido em biblioteca sintética de 890 estruturas válidas:

| Operação | Restaram | Similaridade média | Scaffolds |
| -------- | -------: | -----------------: | --------: |
| seleção por regras | 890 | 0,183 | 101 |
| `--per-scaffold 1` | 101 | — | 101 |
| `--diverse 40` | 40 | 0,147 | 13 |

## Entrega ao SMILES2Docking

`--docking arquivo.xlsx` grava a planilha que o SMILES2Docking lê, com as
colunas `access_code` e `smiles` que ele espera por padrão
(`config/settings.yaml`). Vai o SMILES canônico — a estrutura sobre a qual os
descritores foram calculados — e apenas as moléculas selecionadas, porque
mandar as excluídas para o docking desfaria silenciosamente a seleção.
`--docking-all` entrega todas as moléculas válidas.

## Política recomendada (padrão)

```text
Lipinski: obrigatório, no máximo uma violação
Veber:    obrigatório
Ghose:    informativo
Egan:     informativo
Muegge:   informativo
QED:      calcular e ranquear
PAINS/Brenk: advertir
```

Antes de executar, a interface e a CLI mostram sempre:

> Uma molécula será selecionada quando passar nos perfis obrigatórios. Os perfis
> informativos serão calculados e incluídos no relatório, mas não serão usados
> para exclusão. Alertas PAINS e Brenk não excluirão moléculas.

Exigir aprovação simultânea em todos os perfis não é recomendado: eles foram
desenvolvidos com objetivos, conjuntos de dados e descritores diferentes. Com
quatro ou mais perfis obrigatórios o programa emite um aviso explícito.

---

## Saídas

### Excel

`00_SUMMARY`, `01_SELECTED_FINAL`, `02_EXCLUDED_FINAL`, `03_PROFILE_MATRIX`,
uma aba `FAIL_<PERFIL>` por perfil, abas de alertas e `CONFIG` (perfis, versões,
limites, método de LogP, definição de contagem de átomos, versão do RDKit e
hashes de configuração).

As regras quebradas saem em dois modos:

- **detalhado** — uma aba por código de falha (`LIP_MW_HIGH`, `VEB_TPSA_HIGH`, …);
- **compacto** (`--compact`) — uma única aba `RULE_FAILURES`.

### SQLite

`molecule_descriptors`, `profile_results`, `rule_failures`,
`structural_alerts`, `final_decisions`, `run_config`. As tabelas de falhas e
alertas são esparsas: só registram o que de fato ocorreu.

`profile_results.failure_mask` traz um bitmask das regras quebradas, na ordem
das regras do próprio perfil (bit 0 = primeira regra). Perfis com mais de 63
regras não recebem máscara — `rule_failures` continua sendo o registro
autoritativo.

### Parquet

`--parquet caminho.parquet` grava a tabela larga por registro (mesmas colunas de
`01_SELECTED_FINAL`), preservando os tipos. Útil acima do limite de linhas do
Excel. Requer `pip install 'smiles2select[parquet]'`. Com `--parquet-selected`,
só as moléculas selecionadas.

---

## Arquitetura

```text
DescriptorRegistry → ProfileRegistry → RuleEngine → ScoreEngine → AlertEngine → DecisionEngine
```

```text
src/smiles2select/
├── chemistry/    parsing, padronização, descritores, planejamento,
│                 duplicatas, scaffolds de Murcko, fingerprints
├── profiles/     registry, loader, validator, builtins/*.json
├── rules/        operadores, regras, avaliação vetorizada, explicações
├── scores/       QED, SA score, consenso, ranqueamento
├── alerts/       catálogos RDKit, SMARTS personalizados, políticas
├── decision/     papéis, consenso, parser de expressões, decisão final
├── selection/    MaxMin e limite por scaffold (dependem do conjunto)
├── io/           importação de CSV/TSV/SMI/XLSX
├── storage/      banco SQLite da execução e cache persistente
├── export/       Excel, Parquet e entrega ao SMILES2Docking
├── pipeline/     configuração, workers, orquestração, amostragem, presets
└── gui/          interface PySide6 em sete etapas
```

Perfis são arquivos JSON. Adicionar um perfil ou alterar um limite não exige
tocar no motor:

```json
{
  "id": "meu_perfil",
  "name": "Meu perfil",
  "category": "custom",
  "version": "1.0.0",
  "pass_policy": { "type": "max_violations", "value": 1 },
  "rules": [
    {
      "id": "meu_mw_max",
      "descriptor": "mol_wt",
      "operator": "<=",
      "threshold": 450,
      "severity": "hard",
      "failure_code": "MEU_MW_HIGH",
      "label": "Molecular weight"
    }
  ]
}
```

Coloque o arquivo em `src/smiles2select/profiles/custom/`, ou em qualquer pasta
apontada por `--profile-dir`.

## Presets

Um preset guarda as **decisões** — perfis e seus papéis, política de consenso,
expressão, modo de QED, catálogos e ações de alerta, padronização — e nunca os
arquivos de entrada nem os caminhos de saída. O mesmo preset serve para
qualquer biblioteca, que é o objetivo: o grupo combina uma política e a
reutiliza.

Os perfis são referenciados por id, não copiados. Um preset continua válido
quando um perfil é corrigido, e é **recusado por inteiro** — nunca aplicado pela
metade — se citar um perfil que esta instalação não possui.

Na interface, os botões ficam na tela “5. Política de seleção”; carregar um
preset preserva os arquivos já escolhidos.

## Desempenho

- O planejador calcula apenas os descritores exigidos pelos perfis, escores e
  alertas selecionados.
- Os workers RDKit rodam em processos paralelos (`joblib`), em lotes.
- As regras numéricas são avaliadas de forma vetorizada: uma passagem por regra
  sobre toda a biblioteca, não uma chamada Python por molécula × regra.
- A padronização, a versão do RDKit e o conjunto de descritores entram no hash
  de configuração; mudar qualquer um deles invalida valores em cache.

### Cache persistente (`--cache`)

Guarda, por molécula, o resultado de parsing, padronização, descritores e
alertas. Numa segunda execução sobre biblioteca sobreposta, o trabalho caro é
pulado.

Um valor só é reaproveitado quando **tudo** que poderia alterá-lo é idêntico —
versão do RDKit, perfil de padronização (sais, cargas, tautômeros, estereo) e
catálogos de alerta em vigor entram na chave. O conjunto de descritores é
verificado à parte: uma entrada que não contenha **todos** os descritores
pedidos é recalculada, nunca lida pela metade. Portanto acrescentar um perfil
aproveita o trabalho já feito para os outros, e reduzir os perfis continua
acertando o cache.

Medido em biblioteca sintética de 2000 SMILES (895 estruturas distintas,
5 perfis, PAINS + Brenk, 4 processos), com resultados idênticos nos três modos:

| Execução | Tempo | Cache |
| -------- | ----: | ----- |
| sem cache | 4,63 s | — |
| cache frio | 3,43 s | 0 reaproveitadas, 895 calculadas |
| cache quente | 0,09 s | 895 reaproveitadas, 0 calculadas |

## Testes

```bash
pytest
```

```bash
pytest --cov=smiles2select --cov-report=term-missing
```

A suíte cobre limites exatos de cada perfil (no limite, imediatamente abaixo e
imediatamente acima), múltiplas violações, moléculas aprovadas em um perfil e
reprovadas em outro, alertas com cada ação configurada, políticas de consenso
(2 de 3, 3 de 5, todos, qualquer) e expressões AND/OR/NOT.

## Autoria

**Adriano Marques Gonçalves** — Universidade de Araraquara (UNIARA)

Para citar este software, veja [CITATION.cff](CITATION.cff).

## Licença

MIT — veja [LICENSE](LICENSE).
