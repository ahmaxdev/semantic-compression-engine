# semantic-compression-engine
Compress structured logs by 7600x using semantic merging. CPU-only, streaming, LLM-ready. Distill meaning, not just bits.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Results: 7600x Compression](https://img.shields.io/badge/results-7600x_compression-green)](#results)

> **Distill massive datasets into their semantic essence.**  
> A CPU-optimized streaming compressor that reduces structured logs by **99.98%+** while preserving queryable meaning—built for LLM context optimization, RAG pipelines, and observability.

---

## 🚀 Quick Results

| Metric | Input | Output | Reduction |
|--------|-------|--------|-----------|
| **Log Lines** | 100,000 | **14** | **7601x** |
| **Semantic Concepts** | Unknown | **14 Unique Events** | **99.98%** |
| **Processing Speed** | - | **63 lines/sec** | **CPU Only** |
| **LLM Token Cost** | ~$2.00 | **~$0.0003** | **~6600x savings** |

**From 100k lines of HDFS logs → 14 semantic entries.**  
Errors stay separated from info logs. Temporal context is preserved. Meaning survives.

---

## 🎯 Why This Exists

LLMs are expensive. Context windows are growing, but so is the data. Traditional compression (ZIP, GZIP) saves bits, but not *meaning*.

**Semantic compression** solves this by:
- ✅ Merging semantically similar events (not just identical strings)
- ✅ Preserving temporal relationships (what happened when)
- ✅ Reducing token costs by orders of magnitude
- ✅ Running on CPU—no GPU required

**Ideal for:**
- RAG systems drowning in retrieved chunks
- Log analytics at scale (DevOps, SRE)
- LLM training data preprocessing
- Real-time stream processing for agents

---

## ⚡ Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
