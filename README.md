# Gender-Korpus — Overview

- Corpus analysis of gender assignments in German texts (18th–20th c.) from https://msternchenw.de/. 
- Scans all `*.tsv` with tokens labeled `Mann`, `Frau`, `Genderneutral`, `O`.
- Adds POS tagging for labeled tokens using spaCy (`de_core_news_md`, fallback `de_core_news_sm`).
- Start Dashboard with: Start-Process "\Gender-Korpus\dashboard.html"

## Key totals (last run)

- Tokens total: 684,105
- Labeled total (M/F/N): 29,199
- `Mann`: 17,716 (60.67%)
- `Frau`: 10,200 (34.93%)
- `Genderneutral`: 1,283 (4.39%)
- Other `O`: 654,906

## Outputs (generated files)

- `gender_counts_summary.csv`
- `gender_counts_bar.png`, `gender_counts_donut.png`
- `gender_counts_normalized_per_text.png`, `gender_ratio_per_text.png`
- `gender_diverging_bars_per_text.png`
- `pos_tagged_tokens.csv`, `pos_counts.csv`
- `pos_heatmap.png`
- `pos_distribution_stacked100.png`, `pos_distribution_grouped_top10.png`

## Requirements

- Python 3.12; `pandas`, `numpy`, `matplotlib`, `spacy`
- spaCy model: `de_core_news_md` (auto-fallback to `de_core_news_sm`)
