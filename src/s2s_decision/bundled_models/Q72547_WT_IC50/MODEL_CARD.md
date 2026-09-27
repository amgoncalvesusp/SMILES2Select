# Q72547_WT · IC50

Modelo `gradient_boosting_scalar` para ordenar candidatos apenas na tarefa Q72547_WT/IC50, com rótulo de treinamento pActivity ≥ 6. Usa descritores escalares do contrato químico registrado no manifest. Escore calibrado em partição própria; não representa probabilidade universal de atividade, confirmação experimental nem chance de avanço de projeto.

Treinado com 1.803 registros de treino de Papyrus++ 05.7. O pacote contém somente referências dessa partição. Escolha congelada pela AP de validação (0,9215) entre três baselines antes da análise de teste. Na avaliação retrospectiva posterior, AP do teste filtrado pelo SMILES2Select foi 0,8897; 18/20 ativos na cesta com máximo de três por scaffold. O pool filtrado tinha 120 ativos em 212 candidatos (56,60%). Números dependem dessa tarefa e desse conjunto; não constituem validação externa ou prospectiva.

Manifest e ONNX originais permanecem intactos. O card histórico foi substituído nesta cópia de lançamento porque dizia, antes da avaliação posterior, que o teste retido não havia sido avaliado. Consulte `S2S-Decision/artifacts/product-validation-v1/REPORT.md` e `validation-choice-freeze.json` para protocolo e escolha. Fonte, alterações e licença dos recursos: `../LICENSE_CC_BY_SA_4.0.md`.
