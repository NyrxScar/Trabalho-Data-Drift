# Monitoramento e Pipeline de Alertas de Data Drift

**Evolução do projeto de monitoramento de dados climáticos com geração dinâmica de relatórios.**

| | |
|---|---|
| **Autor** | Nyrx Oliveira de Aquino Farias |
| **Disciplina / Professor** | Trabalho Prático de MLOps · Prof. MSc. Hugo Menezes Barra |
| **Licença** | MIT (ver [LICENSE](LICENSE)) |

Pipeline de MLOps que simula um modelo de previsão de chuva em produção. O pipeline gera dados
climáticos, treina o modelo **uma única vez** e o persiste em `.joblib`. Em seguida, carrega o
artefato apenas para inferência e compara a distribuição dos dados de produção com a de
referência (treino) por meio de testes estatísticos. Ele também mede a qualidade dos dados,
avalia o efeito do drift nas previsões do modelo, recomenda (ou não) o retreinamento e gera
automaticamente um log de auditoria em JSON, um relatório executivo em HTML e um relatório
interativo em Plotly.

### Execução rápida

Requer **Python 3.12 ou superior**. A partir da pasta raiz do projeto, no Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.main
.venv\Scripts\python -m pytest
```

No Linux/macOS, troque `.venv\Scripts\python` por `.venv/bin/python`. O terceiro comando
executa todo o pipeline e o quarto executa os testes automatizados.

Depois, abra no navegador `reports/exec_summary.html` (relatório executivo) e
`reports/drift_report.html` (relatório interativo). Detalhes nas seções
[11](#11-instalação) e [12](#12-execução).

---

## Sumário

1. [Objetivo da atividade](#1-objetivo-da-atividade)
2. [Cenário de monitoramento climático](#2-cenário-de-monitoramento-climático)
3. [Arquitetura e desacoplamento das etapas](#3-arquitetura-e-desacoplamento-das-etapas)
4. [Estrutura de diretórios](#4-estrutura-de-diretórios)
5. [Tecnologias](#5-tecnologias)
6. [Justificativa dos testes estatísticos](#6-justificativa-dos-testes-estatísticos)
7. [Injeção de valores ausentes](#7-injeção-de-valores-ausentes)
8. [Critérios de identificação de drift](#8-critérios-de-identificação-de-drift)
9. [Critérios de retreinamento](#9-critérios-de-retreinamento)
10. [Geração automática de relatórios](#10-geração-automática-de-relatórios)
11. [Instalação](#11-instalação)
12. [Execução](#12-execução)
13. [Testes](#13-testes)
14. [Exemplo dos arquivos gerados](#14-exemplo-dos-arquivos-gerados)
15. [Análise técnica dos resultados](#15-análise-técnica-dos-resultados)
16. [Parecer técnico e ações recomendadas](#16-parecer-técnico-e-ações-recomendadas)
17. [Limitações](#17-limitações)
18. [Possíveis evoluções](#18-possíveis-evoluções)

---

## 1. Objetivo da atividade

- Simular um cenário de produção em que os dados de entrada de um modelo mudam ao longo do tempo.
- Detectar essa mudança (*data drift*) com testes estatísticos adequados a cada tipo de variável.
- Identificar problemas de qualidade (*data quality*) independentemente do drift.
- Medir o efeito dessa mudança no comportamento do modelo em produção (*prediction drift*).
- Transformar os resultados em uma decisão operacional documentada e em relatórios gerados pelo
  código, sem nenhum valor escrito à mão.

## 2. Cenário de monitoramento climático

Um modelo de classificação prevê se **vai chover no dia seguinte** (`rain_tomorrow`). Ele foi
treinado com dados de um período seco e quente (**referência**). Em produção, os dados passam a
vir de um período chuvoso e mais frio (**produção**). Além disso, uma falha intermitente nos
sensores faz com que 5% das leituras de temperatura cheguem vazias.

| Variável | Tipo | Distribuição (referência → produção) |
|---|---|---|
| `temperature` (°C) | numérica | Normal(25, 4) → Normal(22, 4,5) + **5% NaN** |
| `humidity` (%) | numérica | Normal(60, 10) → Normal(75, 10), limitada a [0, 100] |
| `wind_speed` (km/h) | numérica | Weibull(k=2, λ=12) → **igual** (caso de controle, sem drift) |
| `precipitation` (mm) | numérica | Gamma(0,8; 2,5) → Gamma(1,5; 4,0) |
| `sky_condition` | categórica | clear/cloudy/rainy **60/30/10%** → **10/30/60%** |
| `rain_tomorrow` | rótulo (0/1) | só na referência; em produção o *ground truth* ainda não chegou |

- `wind_speed` mantém os mesmos parâmetros de propósito. Assim é possível verificar que o
  pipeline não acusa drift onde ele não existe.
- O rótulo segue uma relação logística com as variáveis (mais umidade, mais precipitação e céu
  `rainy` aumentam a chance de chuva; céu `clear` e temperatura alta diminuem). Ele é sorteado
  com um fluxo aleatório **independente**, então incluí-lo não altera os valores das variáveis
  gerados com a mesma seed.
- O conjunto de produção **não tem rótulo**, como em um sistema real: o resultado ("choveu ou
  não") só é conhecido depois da previsão.

Todos os parâmetros ficam centralizados em [src/config.py](src/config.py).

## 3. Arquitetura e desacoplamento das etapas

Em produção, **treino e inferência são etapas separadas**. Retreinar o modelo a cada execução
desperdiçaria processamento e, pior, faria o modelo "perseguir" os dados de produção. Assim, o
drift deixaria de ser detectado e o comportamento em produção nunca seria auditável. Neste
projeto:

- **Treino** (`src/train_model.py`): roda uma vez, ou quando o retreinamento é aprovado. Salva
  o pipeline ajustado em `models/model.joblib` e os metadados em `models/model_metadata.json`
  (versão, data, métricas, medianas de imputação e o **SHA-256 dos dados de treino**).
- **Inferência** (`src/inference.py`): apenas **carrega** o `.joblib` e chama `predict_proba`.
  Não existe `fit` nesta etapa, e um teste automatizado garante isso.
- **Linhagem:** antes da inferência, o hash do `reference_data.csv` atual é comparado ao hash
  registrado no treino. Se os dados de referência mudaram sem retreino, o pipeline para com erro
  explícito em vez de produzir monitoramento inconsistente.
- **Relatórios** (`src/report_generator.py`): leem o `drift_summary.json` gravado pelo
  monitoramento. Podem ser regenerados sozinhos, sem recalcular nada.

```text
 ┌─────────────── etapa de dados (ingestão simulada) ───────────────┐
 │ generate_data ─► validate ─► reference_data.csv / current_data.csv │
 │                              dataset_metadata.json (seed + SHA-256) │
 └───────────────────────────────┬───────────────────────────────────┘
                                 │ (só se os arquivos não existirem ou --regenerate-data)
 ┌──────────── etapa de treino (execução única) ───────────┐
 │ train_model ─► models/model.joblib + model_metadata.json │
 └───────────────────────────────┬─────────────────────────┘
                                 │ (só se o artefato não existir ou --retrain)
 ┌──────────────── etapa de inferência ─────────────────┐
 │ load_model ─► verify_lineage ─► predict (sem fit)     │
 │ ─► reference_predictions.csv / current_predictions.csv │
 └───────────────────────────────┬───────────────────────┘
 ┌──────────────── etapa de monitoramento ──────────────────────────────┐
 │ detect_drift (KS / Qui-Quadrado / drift de predição) ─► decision       │
 │ (data quality + regra de retreinamento) ─► data/drift_summary.json     │
 └───────────────────────────────┬──────────────────────────────────────┘
 ┌──────────────── etapa de relatórios (lê o JSON) ─────────┐
 │ report_generator ─► exec_summary.html · drift_report.html │
 └──────────────────────────────────────────────────────────┘
```

| Módulo | Responsabilidade |
|---|---|
| `src/config.py` | Caminhos, seed, distribuições, rótulo, modelo, `alpha`, limiares e política de retreinamento |
| `src/exceptions.py` | Exceções de domínio (`DataValidationError`, `ModelArtifactError`) |
| `src/artifacts.py` | SHA-256 de arquivos e leitura/escrita de JSON estrito |
| `src/generate_data.py` | Geração, rótulo, injeção de NaN, validação, CSVs e manifesto |
| `src/train_model.py` | Treino, avaliação em holdout e persistência do modelo (`joblib`) |
| `src/inference.py` | Carga e validação do artefato, verificação de linhagem e predição |
| `src/detect_drift.py` | KS-Test, Qui-Quadrado, drift de predição e métricas de NaN |
| `src/decision.py` | Data quality, consolidação e regra de retreinamento |
| `src/report_generator.py` | JSON de auditoria e relatórios HTML/Plotly gerados a partir do JSON |
| `src/main.py` | Orquestração das etapas e interface de linha de comando |

Os testes estatísticos são executados uma única vez. Cada resultado (`FeatureDriftResult`)
carrega tudo de que os relatórios precisam: estatística, p-value, tamanho de efeito, severidade,
contagens, valores ausentes, frequências observadas e esperadas, graus de liberdade e
interpretação.

## 4. Estrutura de diretórios

```text
.
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── artifacts.py
│   ├── generate_data.py
│   ├── train_model.py
│   ├── inference.py
│   ├── detect_drift.py
│   ├── decision.py
│   ├── report_generator.py
│   └── main.py
├── data/
│   ├── reference_data.csv          (gerado: referência + rótulo)
│   ├── current_data.csv            (gerado: produção, sem rótulo, 5% NaN em temperature)
│   ├── dataset_metadata.json       (gerado: seed e SHA-256 dos datasets)
│   ├── reference_predictions.csv   (gerado: scores do modelo na referência)
│   ├── current_predictions.csv     (gerado: scores do modelo em produção)
│   └── drift_summary.json          (gerado: log de auditoria)
├── models/
│   ├── model.joblib                (gerado: pipeline treinado)
│   └── model_metadata.json         (gerado: versão, métricas e linhagem)
├── reports/
│   ├── exec_summary.html           (gerado: relatório executivo)
│   └── drift_report.html           (gerado: relatório interativo Plotly)
├── tests/
│   ├── conftest.py
│   ├── test_generate_data.py
│   ├── test_detect_drift.py
│   ├── test_decision.py
│   ├── test_model.py
│   └── test_report_generator.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Tecnologias

| Biblioteca | Versão | Uso |
|---|---|---|
| Python | ≥ 3.12 (testado em 3.14.6) | linguagem |
| NumPy | 2.5.4 | geração aleatória reprodutível (`default_rng`) |
| pandas | 3.1.1 | manipulação dos datasets e CSV |
| SciPy | 1.18.1 | `ks_2samp`, `chi2_contingency` |
| scikit-learn | 1.9.1 | pipeline de pré-processamento + `LogisticRegression` |
| joblib | 1.6.0 | persistência do modelo treinado |
| Plotly | 7.1.0 | relatório interativo |
| pytest | 9.2.0 | testes automatizados |

Da biblioteca padrão: `json`, `hashlib`, `logging`, `pathlib`, `dataclasses`, `argparse`,
`html`, `uuid`.

**Modelo:** `Pipeline` do scikit-learn com `SimpleImputer` (mediana) e `StandardScaler` para as
numéricas, `OneHotEncoder` (categorias conhecidas, `handle_unknown="ignore"`) para
`sky_condition`, e `LogisticRegression`. Pré-processamento e estimador ficam no **mesmo
artefato**, o que garante que a inferência aplique exatamente as transformações do treino.

## 6. Justificativa dos testes estatísticos

### KS-Test (variáveis numéricas)

O teste de Kolmogorov-Smirnov de duas amostras (`scipy.stats.ks_2samp`) compara as funções de
distribuição acumulada empíricas das duas amostras. É **não paramétrico** (não supõe
normalidade, o que importa para `wind_speed` e `precipitation`, que são assimétricas) e é
sensível a mudanças de posição, de dispersão e de forma.

- H₀: as duas amostras vêm da mesma distribuição.
- Estatística D: maior distância vertical entre as duas distribuições acumuladas (0 a 1).
  D também é usada como **tamanho de efeito**.
- **Tratamento de NaN:** o teste recebe apenas valores válidos (`dropna`). Os ausentes são
  contados à parte e reportados como métrica de qualidade.
- O mesmo teste é aplicado aos **scores do modelo** (`rain_probability`) na referência e em
  produção, o que mede o *prediction drift*.

### Qui-Quadrado de independência (variável categórica)

KS não se aplica a categorias sem ordem. Para `sky_condition`, monta-se uma tabela de
contingência 2 × k (linhas: referência/produção; colunas: categorias) e aplica-se
`scipy.stats.chi2_contingency`.

- Todas as categorias conhecidas entram na tabela, mesmo com frequência zero em um dos conjuntos.
- Categorias ausentes nos **dois** conjuntos são excluídas do teste (gerariam frequência
  esperada zero) e registradas em `excluded_zero_categories`.
- Categorias desconhecidas (fora de `clear/cloudy/rainy`) entram no teste e geram alerta de
  qualidade.
- Sem correção de Yates (`correction=False`), para que a estatística seja o qui-quadrado de
  Pearson, coerente com o V de Cramér e com o teste por permutação.
- **Condições de aplicação (regra de Cochran):** nenhuma frequência esperada abaixo de 1 e no
  máximo 20% das células abaixo de 5. Se não forem atendidas, o p-value é obtido por
  **permutação Monte Carlo** (10.000 permutações dos rótulos de grupo, preservando as margens).
  O método usado fica registrado em `p_value_method`.
- **Tamanho de efeito:** V de Cramér = √(χ² / (n · (min(linhas, colunas) − 1))).

## 7. Injeção de valores ausentes

`inject_missing_values` sorteia, sem reposição, exatamente `round(0,05 × n)` linhas do conjunto
de **produção** e define `temperature` como `NaN` nelas (250 de 5.000). A função trabalha sobre
uma cópia, e a referência nunca é alterada. `validate_datasets` confirma que a referência não tem
ausentes e que a taxa em produção está dentro de ±1 ponto percentual de 5%.

Na inferência, esses valores são preenchidos pelo `SimpleImputer` com a **mediana do treino**
(24,99 °C). O número de imputações é registrado no JSON e nos relatórios, porque essa imputação
silenciosa é justamente o tipo de falha que o monitoramento precisa tornar visível.

## 8. Critérios de identificação de drift

| Regra | Resultado no relatório |
|---|---|
| p-value < α (padrão 0,05, configurável) | **🚨 DRIFT**: há evidência estatística de diferença |
| p-value ≥ α | **✅ Estável**: sem evidência de drift. Isso *não* prova que as distribuições sejam idênticas |
| menos de 30 valores válidos, menos de 2 categorias ou resultado não finito | **⚠️ Não testado**, com o motivo registrado |

Quando há drift, a **severidade** é classificada pelo tamanho de efeito:

| Métrica | Baixa | Moderada | Alta |
|---|---|---|---|
| KS D | < 0,10 | 0,10 a 0,20 | ≥ 0,20 |
| V de Cramér | < 0,10 | 0,10 a 0,30 | ≥ 0,30 |

A severidade existe porque, com milhares de registros, o p-value fica minúsculo até para
diferenças irrelevantes. A significância estatística sozinha não mede a gravidade.

## 9. Critérios de retreinamento

Três conceitos são avaliados **separadamente**:

1. **Drift estatístico**: resultado dos testes (seção 8).
2. **Data quality**: percentual de ausentes em produção, onde ≥ 1% gera **alerta** e ≥ 10% é
   **crítico**. Categorias desconhecidas também geram alerta.
3. **Retreinamento**: decisão operacional (`RetrainingPolicy` em `config.py`).

**Regra de retreinamento.** Recomendar retreinamento se **qualquer** critério for atendido:

- pelo menos **1** variável com drift de severidade **alta**;
- **≥ 50%** das variáveis com drift;
- **drift de predição** de severidade alta (configurável em
  `prediction_drift_triggers_retraining`).

Problemas de qualidade **nunca disparam o retreinamento sozinhos**, mas mudam a ação:

| Situação | Ação |
|---|---|
| retreinar + qualidade crítica | `FIX_DATA_PIPELINE_BEFORE_RETRAINING` |
| retreinar + qualidade em alerta | `RETRAIN_RECOMMENDED_AFTER_DATA_QUALITY_REVIEW` |
| retreinar, qualidade OK | `RETRAIN_RECOMMENDED` |
| sem retreino, mas com problema de qualidade ou variável não testada | `INVESTIGATE_DATA_QUALITY` |
| drift apenas abaixo dos limiares | `MONITOR` |
| nada encontrado | `NO_ACTION` |

O pipeline **não retreina automaticamente**. Ele emite a recomendação, o nível de alerta e os
motivos (no JSON e no relatório executivo). Depois de aprovado, o retreino é feito
explicitamente com `python -m src.main --retrain` ou `python -m src.train_model`, e gera uma
nova versão do artefato.

## 10. Geração automática de relatórios

1. O monitoramento grava **`data/drift_summary.json`** com timestamp ISO 8601, `run_id`,
   total de variáveis, percentual com drift, estatística, p-value, `has_missing_values` e
   contagem/percentual de NaN por variável, status de data quality, informações do modelo,
   drift de predição, decisão de retreinamento, motivos e critérios. O arquivo é gravado com
   `allow_nan=False`, e `NaN`/`inf` são normalizados para `null`.
2. O `report_generator` **lê esse JSON de volta** (com validação da versão do schema) e gera:
   - **`reports/exec_summary.html`**: faixa de decisão, diagnóstico da saúde dos dados,
     indicadores, destaque da falha de data quality (taxa de NaN e imputação), tabela
     consolidada com status **🚨 DRIFT / ✅ Estável**, seção do modelo em produção,
     diagnóstico por variável e recomendação com motivos. Suporta tema claro e escuro.
   - **`reports/drift_report.html`**: histogramas sobrepostos (referência × produção, mesmos
     bins) das quatro numéricas, proporções de `sky_condition`, percentual de NaN por variável
     com o limiar de alerta, tamanho de efeito por variável, tabela de resultados, distribuição
     dos scores do modelo e taxa prevista de chuva. O Plotly.js vem embutido no arquivo, que
     abre sem internet.

Os mesmos formatadores (`format_statistic`, `format_p_value`) são usados em todos os relatórios,
e os testes verificam que os valores exibidos correspondem aos registrados no JSON.

## 11. Instalação

**Pré-requisito: Python 3.12 ou superior.** As versões das dependências são fixadas em
`requirements.txt` para garantir reprodutibilidade, e algumas delas (como o numpy 2.5.4) não
existem para Python 3.11 ou anterior. O projeto foi desenvolvido e testado em Python 3.14.6.
Os pacotes também estão disponíveis para Python 3.12 e 3.13. Para verificar a versão:
`python --version`.

1. Abra um terminal **na pasta raiz do projeto** (a que contém `src/` e `requirements.txt`).
2. Crie o ambiente virtual e instale as dependências:

   ```bash
   python -m venv .venv
   ```

   **Windows (PowerShell ou CMD):**

   ```powershell
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Se o PowerShell bloquear o `activate` ("execução de scripts foi desabilitada"), dispense a
   ativação e chame o Python do ambiente diretamente, trocando `python` por
   `.venv\Scripts\python` em todos os comandos:

   ```powershell
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

   **Linux/macOS:**

   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 12. Execução

A partir da raiz do projeto, **um único comando** executa todo o pipeline:

```bash
python -m src.main
```

- **1ª execução:** gera os dados, treina e salva o modelo, faz a inferência, monitora e gera
  os relatórios.
- **Execuções seguintes:** reaproveitam os dados e o `model.joblib` (sem treino) e executam
  apenas inferência, monitoramento e relatórios.

Opções e etapas avulsas:

```bash
python -m src.main --retrain                    # retreino aprovado: gera nova versão do modelo
python -m src.main --regenerate-data --retrain  # novos dados (ex.: --seed 7) exigem novo modelo
python -m src.main --alpha 0.01                 # outro nível de significância
python -m src.train_model                       # apenas a etapa de treino
python -m src.report_generator                  # apenas os relatórios, a partir do JSON existente
```

Os comandos usam `python -m`, por isso devem ser executados **a partir da pasta raiz** do
projeto. Os arquivos de entrada e saída, por sua vez, são localizados a partir de
`src/config.py`, e não do diretório atual. Em caso de falha (dados inválidos, artefato ausente,
linhagem divergente), o processo registra o erro no log e retorna código `1`.

**Onde ver os resultados** (cerca de 5 segundos de execução):

| Arquivo | Como abrir |
|---|---|
| `reports/exec_summary.html` | duplo clique, abre no navegador (relatório executivo) |
| `reports/drift_report.html` | duplo clique, abre no navegador (gráficos interativos, funciona offline) |
| `data/drift_summary.json` | qualquer editor de texto (log de auditoria) |
| `models/model_metadata.json` | qualquer editor de texto (versão e métricas do modelo) |

O terminal também mostra o log de cada etapa (`[1/7]` a `[7/7]`) e, ao final, uma linha com o
resumo da execução:

```text
INFO | src.main | [3/7] Training stage.
INFO | src.main | Reusing persisted model artefact in ...\models (no training).
...
INFO | src.main | Run run-...: 4/5 features drifted; prediction drift=drift;
                  action=RETRAIN_RECOMMENDED_AFTER_DATA_QUALITY_REVIEW.
```

> **Segurança:** `joblib.load` desserializa objetos Python (pickle). Carregue apenas artefatos
> gerados pelo próprio pipeline de treino, nunca arquivos de origem desconhecida.

## 13. Testes

```bash
python -m pytest
```

Resultado da última execução: **85 testes aprovados**. Eles cobrem:

- **Geração:** colunas (rótulo só na referência), proporções exatas, 250 NaN apenas em
  `temperature` de produção, coerência do rótulo com as variáveis, independência entre o
  sorteio do rótulo e das variáveis, reprodutibilidade por seed, manifesto com SHA-256,
  falhas de validação.
- **Treino e inferência:** artefatos `.joblib` e metadados persistidos, métricas válidas,
  **inferência sem `fit`** (o teste falha se `Pipeline.fit` for chamado), contagem de
  imputações, predições reproduzíveis após recarregar o arquivo, artefato ausente, linhagem
  divergente, **segunda execução reaproveita o modelo** (o arquivo não é modificado) e dados
  regenerados exigem `--retrain`.
- **Detecção:** igualdade com `ks_2samp`/`chi2_contingency` do SciPy, drift e estabilidade,
  exclusão de NaN com contagem, amostras insuficientes, coluna não numérica, resultado não
  finito, categorias com frequência zero, fallback Monte Carlo, categorias inesperadas,
  dataset vazio, coluna inexistente, `alpha` inválido, arquivos vazios.
- **Decisão:** cada ramo da regra de retreinamento, gatilho de drift de predição (ativado e
  desativado), limiares de qualidade e política configurável.
- **Relatórios:** JSON estrito e igual aos resultados, ida e volta `report → JSON → report`
  sem perda, schema inválido rejeitado, valores e rótulos `🚨 DRIFT / ✅ Estável` no HTML,
  dados do Plotly vindos dos arquivos persistidos e relatórios regenerados apenas a partir do
  JSON.

## 14. Exemplo dos arquivos gerados

Trecho real de `data/drift_summary.json` (segunda execução, que reaproveitou dados e modelo):

```json
{
  "schema_version": "2.0",
  "run": {
    "run_id": "run-20261105T140207Z-8ca0fc79",
    "timestamp": "2026-11-05T14:02:07+00:00",
    "seed": 42,
    "alpha": 0.05,
    "data_generated_this_run": false,
    "model_trained_this_run": false
  },
  "summary": {
    "total_features_analyzed": 5,
    "features_with_drift": 4,
    "drifted_features_pct": 80.0,
    "drifted_features": ["temperature", "humidity", "precipitation", "sky_condition"]
  },
  "features": [
    {
      "feature": "temperature",
      "test": "Kolmogorov-Smirnov (two-sample)",
      "statistic": 0.27795789473684207,
      "p_value": 3.7265571867256956e-166,
      "drift_detected": true,
      "status": "drift",
      "severity": "high",
      "has_missing_values": true,
      "missing_values": {"reference_count": 0, "reference_pct": 0.0,
                         "current_count": 250, "current_pct": 5.0},
      "data_quality_status": "warning"
    }
  ],
  "model": {
    "model_version": "rain-model-20261105T135951Z",
    "lineage_verified": true,
    "imputed_counts": {"temperature": 250},
    "reference_positive_rate": 0.1576,
    "current_positive_rate": 0.9262
  },
  "retraining_decision": {
    "retraining_recommended": true,
    "action": "RETRAIN_RECOMMENDED_AFTER_DATA_QUALITY_REVIEW",
    "alert_level": "critical"
  }
}
```

O `run_id` e o `timestamp` mudam a cada execução. Dados, estatísticas e métricas são idênticos
para a mesma seed. A `model_version` só muda quando o modelo é retreinado.

## 15. Análise técnica dos resultados

Execução com seed 42, α = 0,05 e n = 5.000 registros por conjunto.

### Drift nas variáveis de entrada

| Variável | Teste | Estatística | p-value | Efeito | Status | NaN (produção) |
|---|---|---:|---:|---:|---|---:|
| temperature | KS | 0,2780 | 3,727e-166 | D = 0,2780 | 🚨 DRIFT (alta) | 250 (5,00%) |
| humidity | KS | 0,5484 | < 1e-300 | D = 0,5484 | 🚨 DRIFT (alta) | 0 |
| wind_speed | KS | 0,0126 | 0,8223 | D = 0,0126 | ✅ Estável | 0 |
| precipitation | KS | 0,4520 | < 1e-300 | D = 0,4520 | 🚨 DRIFT (alta) | 0 |
| sky_condition | χ² (gl = 2) | 3571,43 | < 1e-300 | V = 0,5976 | 🚨 DRIFT (alta) | 0 |

"< 1e-300" indica que o SciPy retornou 0,0 por limite de precisão numérica. O valor real é
positivo, porém menor do que um `float` consegue representar.

- **temperature:** média caiu de 24,92 para 22,04 °C e o desvio padrão subiu de 4,00 para 4,54.
  O teste usou os 4.750 valores válidos. Os 250 ausentes (5,00%) foram classificados como
  *alerta* de data quality.
- **humidity:** média subiu de 59,99 para 74,84%. É a maior distância entre as numéricas
  (D = 0,55).
- **precipitation:** média passou de 2,02 para 5,93 mm e a mediana de 1,27 para 4,69 mm, com
  a cauda bem mais longa.
- **sky_condition:** `clear` caiu de 60% para 10%, `rainy` subiu de 10% para 60% e `cloudy`
  ficou em 30%. As condições de Cochran foram atendidas (menor frequência esperada: 1.500),
  então vale a aproximação assintótica.
- **wind_speed:** sem evidência de drift (p = 0,82), como esperado, pois os parâmetros de
  geração são iguais. Isso não prova igualdade, apenas que não houve evidência para rejeitar H₀.

### Modelo em produção

| Indicador | Valor |
|---|---|
| Versão / algoritmo | `rain-model-20261105T135951Z` / LogisticRegression |
| Holdout (1.000 registros da referência) | ROC AUC 0,7946 · acurácia 0,7990 · precisão 0,7552 · recall 0,3942 · F1 0,5180 |
| Taxa real de chuva no holdout | 27,4% |
| Taxa prevista de chuva | **15,8%** (referência) → **92,6%** (produção) |
| Score médio (`rain_probability`) | 0,273 → 0,841 |
| Drift de predição | 🚨 DRIFT · KS D = 0,7754 · p < 1e-300 · severidade alta |
| Imputações na inferência | 250 valores de `temperature` com a mediana de treino (24,99 °C) |
| Linhagem | ✅ hash dos dados de treino confere com `reference_data.csv` |

### Decisão do pipeline

4 de 5 variáveis (80%) com drift, todas de severidade alta, além de drift de predição de
severidade alta. **Os três critérios de retreinamento foram atingidos.** Como `temperature`
está em alerta de qualidade, a ação é `RETRAIN_RECOMMENDED_AFTER_DATA_QUALITY_REVIEW`, com
alerta crítico.

## 16. Parecer técnico e ações recomendadas

**Impacto do drift no modelo.** O modelo aprendeu relações válidas para um clima seco (céu
limpo, pouca chuva, umidade moderada). Em produção, a maior parte das entradas cai em regiões
que eram raras no treino (umidade acima de 80%, precipitação acima de 5 mm, `rainy` em 60% dos
casos). O efeito é visível no próprio modelo: a taxa de previsão de chuva saltou de 15,8% para
92,6%, e a distribuição dos scores praticamente inverteu (KS D = 0,78).

Essa mudança pode ser parcialmente legítima, já que um período mais chuvoso deveria gerar mais
previsões de chuva. Mas o modelo está extrapolando: seus coeficientes foram estimados quase sem
exemplos nessas faixas, e sua calibração (a probabilidade prevista corresponder à frequência
real) não foi validada nelas. Como os rótulos de produção ainda não chegaram, **não é possível
medir a acurácia atual**. Por isso a decisão se baseia em drift de entrada e de predição, que
são os sinais disponíveis antes do *ground truth*.

**Impacto dos dados ausentes.** Os 250 registros sem `temperature` (5%) são preenchidos com a
mediana do treino (24,99 °C), enquanto a temperatura típica atual é cerca de 22 °C. A imputação
insere um valor desatualizado (que, pelo coeficiente negativo da temperatura, puxa a
probabilidade de chuva para baixo nesses registros) e esconde a falha do sensor. Se esses
dados entrassem no próximo treino sem tratamento, o novo modelo aprenderia com valores
artificiais.

**Ações recomendadas (em ordem):**

1. **Corrigir/investigar a falha dos sensores de temperatura** e definir a política para os
   registros afetados (descartar, imputar com estatística do período atual ou recuperar na
   origem).
2. **Coletar os rótulos reais** (`rain_tomorrow`) do período de produção, à medida que forem
   conhecidos, e medir o desempenho real do modelo atual (acurácia, recall, calibração).
3. **Retreinar** com dados que incluam o novo regime climático
   (`python -m src.main --retrain`), validar a nova versão no holdout e só então promovê-la.
4. **Manter o monitoramento** a cada novo lote. Com o modelo novo, a próxima execução deve
   comparar a produção com a nova referência.

## 17. Limitações

- Os dados são sintéticos e as variáveis são geradas de forma independente. Em dados reais,
  `sky_condition`, `humidity` e `precipitation` são correlacionadas.
- Os testes são univariados e não detectam mudanças apenas na relação entre variáveis
  (drift multivariado).
- Os rótulos de produção não existem nesta simulação, então não se mede *concept drift* nem a
  queda real de desempenho. O retreino com dados de produção rotulados fica como etapa
  operacional futura: `train_model` treina sobre a referência disponível.
- O modelo é treinado uma única vez e versionado por arquivo. Não há *model registry* (como
  MLflow) nem promoção controlada entre versões.
- A distribuição de scores de referência é calculada sobre toda a referência, incluindo os
  80% usados no treino. Para comparar distribuições isso é aceitável, mas não serve como
  medida de desempenho.
- Com amostras grandes, o KS e o χ² detectam diferenças mínimas. A severidade mitiga isso, mas
  os limiares (0,10/0,20 e 0,10/0,30) são convenções e devem ser calibrados no domínio.
- Não há correção para múltiplas comparações (5 testes com o mesmo α). Uma correção de
  Bonferroni (α/5 = 0,01) não mudaria as conclusões desta execução, pois os p-values com drift
  são muito menores e o de `wind_speed` é 0,82.
- O KS assume distribuições contínuas. Os valores arredondados a duas casas geram empates, que
  tornam o teste levemente conservador.
- O relatório Plotly fica com cerca de 5 MB porque embute o Plotly.js e os dados brutos, para
  funcionar sem internet.

## 18. Possíveis evoluções

- Registro de modelos e experimentos (MLflow) com promoção controlada (staging → produção).
- Avaliação de desempenho real quando os rótulos de produção chegarem (*concept drift*).
- Execução em janelas temporais e acompanhamento histórico do drift (séries de JSONs).
- Métricas complementares (PSI, distância de Wasserstein, Jensen-Shannon).
- Correção para múltiplas comparações (Bonferroni/Holm), configurável.
- Detecção de drift multivariado (por exemplo, classificador referência × produção).
- Alertas automáticos (e-mail, Slack, webhook) e agendamento (cron, Airflow, GitHub Actions).
- Retreinamento acionado pela recomendação, com aprovação humana e validação antes da promoção.
