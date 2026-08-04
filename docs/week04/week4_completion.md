# Week 4 Completion Record

## Final Test Baseline

- Bailian Provider tests: 5 passed
- Full test suite: 267 passed in 3.00s
- OpenAI Python SDK: 2.50.0
- LLM Provider: Bailian OpenAI Compatible
- LLM Model: qwen3.7-plus
- Thinking Mode: disabled
- Structured Output: JSON Object

## Chunk Baseline

### Midea 2024

- Fixed Length 800/120:
  - Dataset ID:
    `chunk_dataset_midea_group_2024_d27f3c7919cfba40f995b86a`
  - Chunk Count: 491

- Paragraph:
  - Dataset ID:
    `chunk_dataset_midea_group_2024_57d7395e75a1d7e479644fca`
  - Chunk Count: 528

- Section Paragraph:
  - Dataset ID:
    `chunk_dataset_midea_group_2024_f2315ec326e8e8b915f19145`
  - Chunk Count: 967

### Hisense 2024

- Fixed Length:
  - Dataset ID:
    `chunk_dataset_hisense_home_2024_e4037c657b0225201366f46b`
  - Chunk Count: 442

- Paragraph:
  - Dataset ID:
    `chunk_dataset_hisense_home_2024_1f9b18e4dfd5d184baa19baa`
  - Chunk Count: 442

- Section Paragraph:
  - Dataset ID:
    `chunk_dataset_hisense_home_2024_af4c817c8d1ce908d1186fc0`
  - Chunk Count: 1035

## Dense Vector Index

- Embedding Model:
  `BAAI/bge-small-zh-v1.5`
- Dimension: 512
- Search: Exact Cosine
- Runtime: CPU

### Midea

`vector_index_midea_group_2024_866466bf6366ad4e6dc50dc9`

### Hisense

`vector_index_hisense_home_2024_25d43849316e318af26c9094`

## Retrieval Smoke Results

### Midea 5 Cases

| Strategy | R@1 | R@3 | R@5 |
|---|---:|---:|---:|
| Fixed Length | 0.60 | 0.60 | 0.80 |
| Paragraph | 0.60 | 0.60 | 0.60 |
| Section Paragraph | 0.80 | 0.80 | 0.80 |

These results are smoke-test results and do not represent
overall system performance.

## Known Dense Retrieval Failure

Hisense 2024 consolidated inventory:

- Gold PDF Page: 112
- Printed Page: 110
- Value: 7,566,932,954.39 CNY
- Fixed Rank: 24
- Paragraph Rank: 24
- Section Paragraph Rank: 23

Primary failure:

`dense_exact_metric_ranking`

Contributors:

- short ambiguous metric
- missing statement section
- high-frequency financial-note interference

## Trusted Answer Acceptance

### Midea 2024 Revenue

- Final status: answered
- Provider request count: 1
- Generator ID: bailian_openai:qwen3.7-plus
- Citation IDs: E1
- Used Chunk:
  `chunk_midea_group_2024_a440b106dbc73c66de0e5303`
- Final answer:
  `美的集团2024年营业收入为407,149,600千元 [E1]。`

### Hisense 2024 Inventory

- Final status: refused
- Provider request count: 0
- Generator ID: none
- Citation IDs: none
- Refusal reason:
  `Top-k 证据未在同一个 Chunk 中同时提供目标报表标题，以及与目标指标邻近且可归属的财务数值`

## Trust Guarantees

- No cross-page Chunk.
- Page ID is the primary page identity.
- Metadata filtering is applied before vector ranking.
- Evidence must come from the active report document.
- Only supporting evidence enters generation context.
- Insufficient evidence prevents model invocation.
- Generated citations must be in the allowed citation list.
- Inline citations must match structured citation IDs.
- API credentials are injected through environment variables.