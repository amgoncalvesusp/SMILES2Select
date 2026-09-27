# Qualificação dos modelos locais para catálogo

Inventário realizado em 26 de setembro de 2026 sobre `S2S-Decision/artifacts/product-validation-v1`. Este registro separa **qualificação dos recursos** de **verificação do instalador**. Três modelos pré-escolhidos por validação foram copiados para `src/s2s_decision/bundled_models/`, com manifests e ONNX históricos intactos.

## Resultado

Há 12 pacotes ONNX em `models/`: três tarefas, com logística, gradient boosting escalar, gradient boosting escalar+Morgan e Tiny por tarefa. Todos contêm `manifest.json`, `model.onnx`, `MODEL_CARD.md` e referências de treino. A inspeção `s2s_decision.decision.list_models()` aprovou os 12 no ambiente local atual, incluindo schema, hashes do ONNX e das referências, tarefa, calibração e contrato químico. Os hashes de manifest e ONNX de cada pacote coincidem com `validation-choice-freeze.json`.

As três opções abaixo foram escolhidas no conjunto de **validação**, antes da leitura das previsões de teste. São os recursos qualificados para distribuição, sempre vinculados ao alvo, endpoint e limiar declarados. O catálogo não escolhe uma delas automaticamente para uma biblioteca do usuário.

| Tarefa | Candidato congelado | Manifest SHA-256 | ONNX SHA-256 | Referências de treino |
|---|---|---|---|---:|
| Q72547_WT / IC50 / pActivity ≥ 6 | `gradient_boosting_scalar` | `d2c526617dce41e68823c20497497150a8410da2639a5501191d57315ef59dec` | `6647342a920b7b7410697ca71ffe1dd1460238449c94875dbc189f9a13847b1c` | 1.803 |
| P0DMS8_WT / Ki / pActivity ≥ 6 | `logistic_scalar_morgan` | `494d4203fdc6103b60fe177aeaa9f5e9185e8c6570c5c4e0450bdf610a215227` | `f1334c10f3dbed984784ec7b3e1490c8c7d2fa46d48f9ba5fb1bab250e3bedee` | 1.000 |
| Q07869_WT / EC50 / pActivity ≥ 6 | `logistic_scalar_morgan` | `fc5b7a1aa7382645e0e4fb52abd394b5778ff5cf95193b69d8ac0a1f0ad96912` | `0267bd1f9b1e71660c91f3d9f527bda526417b9258afc5f58b0a219109870a3d` | 345 |

Os três compartilham o hash químico histórico `a2a3bbe752d2485920b99afd4b3338ab1d0654b8c39894898b1fd04725776923` na versão 3.3.1 e RDKit 2026.03.5. A origem registrada é Papyrus++ 05.7, arquivo consolidado SHA-256 `8004e0d1027a760f205b45264386f792e7d49658da39f77f52e660a6f19760dd`; `source-extraction.json` liga os recortes de cada alvo aos respectivos hashes. `plan.json` define exclusões, divisão por scaffold, seed 42 e critério de escolha. `validation-choice-freeze.json` registra os candidatos antes da avaliação de teste. `REPORT.md` descreve resultados e limitações; são evidências retrospectivas internas, sem validação prospectiva ou externa por outra fonte.

## Licença e composição do pacote

O [registro oficial do Papyrus 05.7 no Zenodo](https://zenodo.org/records/13987985) lista o arquivo `05.7++_combined_set_without_stereochemistry.tsv.xz` com MD5 `9a325b965fb2ab043679d974d1f79549`; MD5 local coincide. Metadados da [API oficial do registro](https://zenodo.org/api/records/13987985) declaram `metadata.license.id = cc-by-sa-4.0`. O `LICENSE.txt` contido em `05.7_additional_files.zip` repete a licença CC BY-SA 4.0; hash MD5 do ZIP coincide com o exibido no registro. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) permite redistribuição e adaptação sob suas condições de atribuição e CompartilhaIgual.

Os três recursos de modelo e referências têm [aviso de atribuição](../src/s2s_decision/bundled_models/LICENSE_CC_BY_SA_4.0.md) e [texto integral da licença](../src/s2s_decision/bundled_models/LICENSE_CC_BY_SA_4.0.txt) separados da licença MIT do código. Foram copiados apenas `manifest.json`, `model.onnx` e `references/{manifest.json,records.jsonl}`. Cada pasta recebeu novo `MODEL_CARD.md`, que relata a avaliação posterior e corrige a cronologia do card histórico sem alterar a origem. Os 3.148 registros de referência correspondem exatamente às partições `train` das três tarefas; nenhum registro `validation`, `calibration` ou `test` consta nessas referências. Os nove pacotes não escolhidos permanecem comparadores da avaliação; Tiny mantém status experimental. Resultado de teste não alterou escolhas congeladas.

## Compatibilidade e verificação

`list_models()` aprovou os três recursos copiados sob SMILES2Select 3.4.0 e RDKit 2026.03.5, incluindo integridade e contrato químico legado validado pelo runtime atual. Uma prévia real com ONNX Runtime CPU e o modelo `Q72547_WT_IC50` pontuou os 228 candidatos da biblioteca sem rótulos e produziu cesta de 20 com máximo de três por scaffold. Esse teste confirma execução local; não demonstra desempenho externo. Testes automatizados conferem os três hashes congelados, cards, licença, descoberta e referências restritas a treino.

O pacote 3.4.0 fixa RDKit em 2026.03.5: os modelos existentes usam exatamente essa versão no contrato químico. A [versão 2026.3.5 no PyPI](https://pypi.org/project/rdkit/2026.3.5/) oferece wheels Python 3.11 para Windows e Linux. Versões posteriores exigem nova verificação de compatibilidade química; não se muda o hash histórico por conveniência.

O wheel 3.4.0 e os bundles Windows/Linux incluem os três pacotes, cards, referências e licença separada. Em cada executável congelado, um teste de inferência real pontuou candidatos com cada um dos três modelos; nenhum deles exigiu dependências de treinamento no bundle. O instalador Windows foi compilado a partir do bundle verificado. Esse smoke confirma execução das tarefas distribuídas, sem medir eficácia prospectiva.

Fontes locais: `S2S-Decision/artifacts/product-validation-v1/{plan.json,source-extraction.json,validation-choice-freeze.json,REPORT.md,models/}`. Nenhum arquivo do estudo foi modificado nesta qualificação.
