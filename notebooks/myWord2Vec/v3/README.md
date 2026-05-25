# v3 — Word2Vec WikiText-103 (vocab=10K)

## Diferencias con v2

| Aspecto | v2 (gaianet/london) | v3 (wikitext-103) |
|---------|:-------------------:|:------------------:|
| Dataset | gaianet/london | iohadrubin/wikitext-103-raw-v1 |
| Rows | 661 | 29,567 |
| Total chars | ~187K | ~519M |
| Vocab | 3,291 (c≥2) | 10,000 (top 10K, c≥5) |
| Embed dim | 64 | **128** |
 | Negative sampling | loop Python | vectorizado (NumPy) |
| Create pairs | doble loop Python | vectorizado (NumPy slices) |
| Embed dim | 64 | **128** |

## Cambios de implementación

1. **Dataset masivo** — Wikitext-103 es ~2,780× más grande que gaianet/london. Para evitar OOM, se procesan solo los primeros 10M de caracteres en lugar de cargar los 519MB completos.

2. **Vocab acotado** — Se toman las 10K palabras más frecuentes con frecuencia ≥ 5 (`most_common(10000)`). Esto mantiene el softmax manejable sin perder cobertura significativa.

3. **Create pairs vectorizado** — v2 iteraba cada palabra con doble loop Python (`for i, word: for j in range(...)`) haciendo 2 dict lookups por par. v3 pre-convierte todo a índices NumPy y construye contextos con slices:
   - `np.array([text_vocab.get(w, -1) for w in words])` — un solo pass
   - Slices `word_ids[left:i]` y `word_ids[i+1:right]` — sin condición por par
   - Ganancia estimada: **~5–10× más rápido**

4. **Negative sampling vectorizado** — v2 iteraba cada sample en Python (`for target_idx in train_context: np.random.choice(...)`), lo cual es inviable con millones de pares. v3 usa:
   - `np.random.randint(0, vocab_size, size=(N, 5))` — genera todos los índices en una operación C
   - `np.repeat()` + `.ravel()` — expande targets y aplana en operaciones vectorizadas
   - Ganancia estimada: **~100–1000× más rápido**

4. **Salida** — Guarda en `target_embeddings.npy`, `context_embeddings.npy`, `text_vocab.npy` (dentro de `v3/`).
