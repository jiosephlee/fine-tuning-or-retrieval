# Human Blog Quality Report vs DPO

## DPO Baseline (reference)
- Mean words: 4820.0
- Mean headings: 14.7
- Mean equation markers: 34.3
- Mean boilerplate noise markers: 0.0

## Per-file comparison

| File | Words | Headings | Eq markers | Noise | Avg sent len | DPO-style score | Band |
|---|---:|---:|---:|---:|---:|---:|---|
| `data/arxiv/explanations/ByteLatent/human/human_blog_1.txt` | 2928 | 37 | 3 | 9 | 19.3 | 0.94 | moderately_close |
| `data/arxiv/explanations/ByteLatent/human/human_blog_2.txt` | 3089 | 30 | 0 | 4 | 20.6 | 0.83 | moderately_close |
| `data/arxiv/explanations/ByteLatent/human/human_blog_3.txt` | 2583 | 11 | 0 | 0 | 24.4 | 0.53 | weaker |
| `data/arxiv/explanations/FeatLLM/human/human_blog_1.txt` | 1426 | 9 | 0 | 0 | 32.4 | 0.39 | weaker |
| `data/arxiv/explanations/FeatLLM/human/human_blog_2.txt` | 1890 | 7 | 0 | 0 | 45.0 | 0.4 | weaker |
| `data/arxiv/explanations/FeatLLM/human/human_blog_3.txt` | 2190 | 25 | 0 | 0 | 27.0 | 0.74 | moderately_close |
| `data/arxiv/explanations/GSPO/human/human_blog_1.txt` | 985 | 10 | 2 | 0 | 29.0 | 0.38 | weaker |
| `data/arxiv/explanations/GSPO/human/human_blog_2.txt` | 1935 | 19 | 0 | 1 | 18.4 | 0.56 | weaker |
| `data/arxiv/explanations/GSPO/human/human_blog_3.txt` | 5274 | 35 | 1 | 0 | 34.7 | 1.2 | close_or_stronger |
| `data/arxiv/explanations/LongRoPE/human/human_blog_1.txt` | 2882 | 27 | 16 | 0 | 22.9 | 0.93 | moderately_close |
| `data/arxiv/explanations/LongRoPE/human/human_blog_2.txt` | 1083 | 30 | 0 | 0 | 32.8 | 0.72 | moderately_close |
| `data/arxiv/explanations/LongRoPE/human/human_blog_3.txt` | 505 | 7 | 0 | 0 | 11.2 | 0.27 | weaker |
| `data/arxiv/explanations/fa3/human/human_blog_1.txt` | 1817 | 10 | 0 | 0 | 21.4 | 0.45 | weaker |
| `data/arxiv/explanations/fa3/human/human_blog_2.txt` | 2143 | 14 | 0 | 0 | 24.4 | 0.54 | weaker |
| `data/arxiv/explanations/fa3/human/human_blog_3.txt` | 1234 | 9 | 0 | 0 | 26.3 | 0.37 | weaker |
| `data/arxiv/explanations/xLSTM/human/human_blog_1.txt` | 2934 | 13 | 0 | 0 | 28.5 | 0.6 | weaker |
| `data/arxiv/explanations/xLSTM/human/human_blog_2.txt` | 2410 | 11 | 0 | 0 | 23.4 | 0.52 | weaker |
| `data/arxiv/explanations/xLSTM/human/human_blog_3.txt` | 2072 | 16 | 83 | 0 | 24.7 | 1.04 | close_or_stronger |

## Domain-level summary

### ByteLatent
- Mean DPO-style score: 0.77
- close_or_stronger: none
- moderately_close: human_blog_1.txt, human_blog_2.txt
- weaker: human_blog_3.txt

### FeatLLM
- Mean DPO-style score: 0.51
- close_or_stronger: none
- moderately_close: human_blog_3.txt
- weaker: human_blog_1.txt, human_blog_2.txt

### GSPO
- Mean DPO-style score: 0.71
- close_or_stronger: human_blog_3.txt
- moderately_close: none
- weaker: human_blog_1.txt, human_blog_2.txt

### LongRoPE
- Mean DPO-style score: 0.64
- close_or_stronger: none
- moderately_close: human_blog_1.txt, human_blog_2.txt
- weaker: human_blog_3.txt

### fa3
- Mean DPO-style score: 0.45
- close_or_stronger: none
- moderately_close: none
- weaker: human_blog_1.txt, human_blog_2.txt, human_blog_3.txt

### xLSTM
- Mean DPO-style score: 0.72
- close_or_stronger: human_blog_3.txt
- moderately_close: none
- weaker: human_blog_1.txt, human_blog_2.txt
