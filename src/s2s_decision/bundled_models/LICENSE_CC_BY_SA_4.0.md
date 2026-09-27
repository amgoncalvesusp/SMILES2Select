# Licença dos modelos distribuídos

Os arquivos em `bundled_models/Q72547_WT_IC50/`, `bundled_models/P0DMS8_WT_Ki/` e `bundled_models/Q07869_WT_EC50/` — pesos ONNX, manifests, referências de treino e cards — são distribuídos sob [Creative Commons Atribuição-CompartilhaIgual 4.0 Internacional (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/legalcode). A licença MIT do código SMILES2Select permanece separada.

**Fonte e crédito:** Béquignon, Olivier J. M.; Schoenmaker, Linde. *Dataset - Papyrus 2024 - A large scale curated dataset aimed at bioactivity predictions*, versão 2024.2/05.7, [Zenodo DOI 10.5281/zenodo.13987985](https://doi.org/10.5281/zenodo.13987985). Arquivo usado: `05.7++_combined_set_without_stereochemistry.tsv.xz` (MD5 `9a325b965fb2ab043679d974d1f79549`). O [registro oficial do Zenodo](https://zenodo.org/records/13987985) declara licença CC BY-SA 4.0.

**Alterações:** seleção de três alvos e endpoints; exclusão de identidades usadas em estudos anteriores; preparação de propriedades e fingerprints; divisão por scaffold em treino, validação, calibração e teste; treinamento e calibração dos modelos; exportação para ONNX; retenção exclusiva das referências de treino. Protocolo, hashes e resultados: `S2S-Decision/artifacts/product-validation-v1/{plan.json,source-extraction.json,validation-choice-freeze.json,REPORT.md}`. Os cards nesta pasta descrevem avaliação posterior à geração dos cards históricos. O conjunto original do Zenodo não endossa estes modelos nem suas previsões.

Ao redistribuir ou adaptar estes recursos, manter atribuição, link para licença e indicação das alterações. Adaptações dos recursos devem respeitar os termos de CompartilhaIgual da licença acima.
