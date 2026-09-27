# Candidato de integração SMILES2Select 3.4.0

Data: 26 de setembro de 2026. Base do repositório canônico: `b9f00bd9699f412f7c1cc1e3eaec436f46bc5d4b` (3.3.1). Origem do motor incorporado e hashes dos arquivos: [INTEGRATION_SOURCE_MANIFEST.json](INTEGRATION_SOURCE_MANIFEST.json). Este candidato não é uma publicação nem altera o estudo/manuscrito.

## Produto entregue

Uma instalação contém `smiles2select` e `s2s_decision`. A seleção química continua sendo o fluxo padrão. O workspace oferece prévia por modelo, comparação e adoção explícita na mesma cesta, com undo/redo, sessão SQLite completa e exportação da seleção adotada. O treinamento e a avaliação continuam nas ferramentas avançadas, com runtime Python externo configurado pelo usuário. Três modelos específicos de alvo, congelados por validação, estão incluídos; Tiny permanece experimental e não foi promovido a padrão. [Qualificação, tarefas, proveniência e licenças](MODEL_CATALOG_QUALIFICATION.md).

O wheel usa Python ≥3.11 e RDKit 2026.3.5. Inferência por código exige o extra `inference`; sem ele, o catálogo mostra a indisponibilidade e a seleção química continua. Os bundles incluem ONNX Runtime CPU. A carga ONNX executa exatamente os bytes cuja soma SHA-256 foi verificada; dados externos ao arquivo ONNX não são aceitos nesse caminho.

## Verificação

| Verificação | Resultado |
|---|---|
| Suíte conjunta de código, Qt e motor Decision | 938 passaram; 3 ignorados; 113 avisos de depreciação em bibliotecas externas |
| Cobertura com branches dos quatro módulos centrais de integração | 84% total; catálogo 87%, adaptador 89%, painel 80%, sessão 84% |
| Regressão adicional: seleção com 5.001 registros, salvar/reabrir/exportar | Passou; scaffolds e planilha final idênticos |
| Lint Ruff `src tests` | Passou |
| Wheel 3.4.0 instalado fora dos repositórios | Ambos os namespaces importados do `site-packages`; três modelos descobertos |
| Executáveis congelados Windows e Linux | 4 testes por plataforma: worker e inferência real nos três modelos distribuídos |
| Auditoria de dependências nativas Windows | 3.724 arquivos, 510 nativos; todas as importações não sistêmicas incluídas |
| Instalador Windows Inno Setup 6.7.3 | Compilação passou a partir do bundle final |
| `pip-audit` do projeto, dependências padrão | Nenhuma vulnerabilidade conhecida no conjunto resolvido |
| `pip-audit` de ONNX Runtime 1.20.1 no bundle Windows | Nenhuma vulnerabilidade conhecida para esse pacote; não substitui auditoria de todas as dependências opcionais |

Comparação local com o relatório preservado de 3.3.1 em `build/gui-selection-v3.3.1/report.json`: mesmos 2.000 IDs finais e mesmo SHA-256; duas seleções/mapas passaram de 3,544/3,479 s para 3,148/2,996 s. RSS final passou de 677,1 para 725,9 MiB. Entrada: 203.390 linhas de teste geradas pela repetição de seis SMILES reais. Essa comparação mede regressão de software; não mede throughput de 203.390 estruturas químicas únicas nem desempenho biológico.

## Artefatos locais e SHA-256

| Artefato | SHA-256 |
|---|---|
| `dist/smiles2select-3.4.0-py3-none-any.whl` | `65baa00fd266fecfb607962195fda3f101e51358172f209137447b8cf840faa3` |
| `dist/SMILES2Select-Setup-3.4.0.exe` | `ba62e3688726c298f9bbbf030cac63c1fdf33100f0589519950a4e7442d37f19` |
| `dist/SMILES2Select-linux-3.4.0.tar.gz` | `9592f973657792f0d7573d29432c16b99fee0e7aec1118128e94bfce9bf491b3` |
| `dist/SMILES2Select/SMILES2Select.exe` | `0d0aaa20e478870e3531bae63b31a8399d28e5c609f00fb5d48311887e810831` |
| `dist/SMILES2Select-manifest.json` | `a20682154f186e0e796c572633f762e233cd8d8f6930ecb519c45d3f3dca50d5` |

O instalador foi compilado e o bundle executou inferência; instalação, atualização e desinstalação em uma máquina Windows limpa não foram executadas neste ambiente. O script instala o programa sob `LocalAppData/Programs`, enquanto modelos importados usam o diretório de dados do usuário; essa separação protege os dados, mas não substitui teste de atualização. Nenhuma tag nem publicação foi criada.
