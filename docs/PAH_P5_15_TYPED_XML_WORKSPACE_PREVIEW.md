# P5-15 — Optional XML workspace preview increment

The XML structure view now offers a separate, opt-in typed preview. It uses a signed, existing XML source token and a bounded POST to the existing ML_Lab XML ingestion/parsing authority. Default C1–C7 extraction selection, recipes, preview, materialization, and provenance routes are unchanged.

Select one record root, 1–12 variables and up to 10 records. The server imposes a 2,000-node bound per representation. The browser shows the original versioned record preview JSON with lineage and missingness; values remain strings. No data is persisted, encoded, or shared with PAH or another module; browser output is in-memory only.

Known limitations: this does not synchronize with the legacy C1–C7 record-root selector, provide linked-child preview in the UI, establish durable record IDs, or materialize typed artifacts. All require separate reviewed contracts. Source fingerprint verification still occurs before returning content. No cache is permitted on the response.
