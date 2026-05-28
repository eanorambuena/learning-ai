# Changelog

## v3 → v4

| Aspecto | v3 (wikitext-103) | v4 (wikitext-103, vocab grande) |
|---------|:------------------:|:-------------------------------:|
| Max chars usados | 300K | **1M** |
| Filter `len(w) > 1` | ✅ (elimina "a", "i") | ❌ **solo filtrar vacíos** |
| `most_common(K)` | 10K | **25K** |
| Vocab resultante | 3,904 | **~10-12K** (estimado) |
| Embedding params | 499,712 | **~1.5M** (target + context) |
| Secuencias retenidas en nb23 | ~10-15% del texto | **~35-50%** (estimado) |

**Motivo:** El vocab v3 de 3,904 palabras era el cuello de botella de nb23. Incluso sin el filtro `len>2`, ~85% del texto quedaba OOV, rompiendo la coherencia de las secuencias. Con v4 se espera que nb23 retenga la mayoría del texto, mejorando significativamente la accuracy.

### Cambios en el notebook (`15_v4_word2vec_largevocab.ipynb`)

| Línea | V3 | V4 |
|:-----:|:--:|:--:|
| 102 | `MAX_CHARS = 300_000` | `MAX_CHARS = 1_000_000` |
| 111-113 | `words = [w for w in words if len(w) > 1]` | **solo filtrar vacíos** (`if w`) |
| 134 | `most_common(10000)` | `most_common(25000)` |

**Impacto estimado:**
- Vocab: 3,904 → **~10-12K palabras**
- Secuencias retenidas en nb23: ~10-15% del texto → **~35-50%**
- Embedding (target+context): 499K params c/u → **~1.5M params c/u**
- Tiempo de entrenamiento: ~30 min → **~1h** (CPU, ~6 min/epoch)

---

## v2 → v3

| Aspecto | v2 (gaianet/london) | v3 (wikitext-103) |
|---------|:-------------------:|:------------------:|
| Dataset | gaianet/london | iohadrubin/wikitext-103-raw-v1 |
| Rows | 661 | 29,567 |
| Total chars | ~187K | ~519M |
| Max chars usados | 187K | **300K** |
| Vocab | 3,291 (c≥2) | **3,904** (top 10K, c≥2) |
| Embed dim | 64 | **128** |
| Create pairs | doble loop Python | vectorizado (NumPy slices) |
| Negative sampling | loop Python | vectorizado (NumPy) |
| Batch size | 256 | **64** |
| Shuffle buffer | len(all_words) | **10,000** |

### Cambios de implementación (v2 → v3)

1. **Dataset masivo** — Wikitext-103 es ~2,780× más grande que gaianet/london. Para evitar OOM, se procesan solo los primeros 300K caracteres (~1.6× más que v2). Con 300K chars se generan ~150K pares positivos + 750K negativos (~12K steps/epoch con batch=64).

2. **Memory fit optimizations** — Para evitar warnings de `cpu_allocator_impl`:
   - `batch_size`: 256 → **64**
   - `shuffle buffer`: `len(all_words)` → **10,000** (no carga todo el dataset en shuffle)

3. **Early Stopping** — `monitor='loss', patience=2, restore_best_weights=True`. Corta automáticamente cuando la loss deja de mejorar (~2-3 épocas).

4. **Vocab acotado** — Se toman las 10K palabras más frecuentes con frecuencia ≥ 2 (`most_common(10000)`, `c >= 2`). Vocab resultante: 3,904 palabras — supera las 3,291 de v2.

5. **Create pairs vectorizado** — v2 iteraba cada palabra con doble loop Python (`for i, word: for j in range(...)`) haciendo 2 dict lookups por par. v3 pre-convierte todo a índices NumPy y construye contextos con slices:
   - `np.array([text_vocab.get(w, -1) for w in words])` — un solo pass
   - Slices `word_ids[left:i]` y `word_ids[i+1:right]` — sin condición por par
   - Ganancia estimada: **~5–10× más rápido**

6. **Negative sampling vectorizado** — v2 iteraba cada sample en Python (`for target_idx in train_context: np.random.choice(...)`), lo cual es inviable con millones de pares. v3 usa:
   - `np.random.randint(0, vocab_size, size=(N, 5))` — genera todos los índices en una operación C
   - `np.repeat()` + `.ravel()` — expande targets y aplana en operaciones vectorizadas
   - Ganancia estimada: **~100–1000× más rápido**

7. **Salida** — Guarda en `target_embeddings.npy`, `context_embeddings.npy`, `text_vocab.npy`.
