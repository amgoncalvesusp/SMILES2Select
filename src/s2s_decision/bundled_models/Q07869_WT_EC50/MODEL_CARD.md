# Q07869_WT · EC50

Modelo `logistic_scalar_morgan` para ordenar candidatos apenas na tarefa Q07869_WT/EC50, com rótulo de treinamento pActivity ≥ 6. Usa descritores escalares e fingerprint Morgan do contrato químico registrado no manifest. Escore calibrado em partição própria; não representa probabilidade universal de atividade, confirmação experimental nem chance de avanço de projeto.

Treinado com 345 registros de treino de Papyrus++ 05.7. O pacote contém somente referências dessa partição. Escolha congelada pela AP de validação (0,9006) entre três baselines antes da análise de teste. Na avaliação retrospectiva posterior, AP do teste filtrado pelo SMILES2Select foi 0,7334; 15/20 ativos na cesta com máximo de três por scaffold. O pool filtrado tinha 41 ativos em 56 candidatos (73,21%). A amostra de teste é pequena; não há validação externa ou prospectiva.

Manifest e ONNX originais permanecem intactos. O card histórico foi substituído nesta cópia de lançamento porque dizia, antes da avaliação posterior, que o teste retido não havia sido avaliado. Consulte `S2S-Decision/artifacts/product-validation-v1/REPORT.md` e `validation-choice-freeze.json` para protocolo e escolha. Fonte, alterações e licença dos recursos: `../LICENSE_CC_BY_SA_4.0.md`.
