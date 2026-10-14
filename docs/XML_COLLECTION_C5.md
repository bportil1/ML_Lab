# XML-C5: XML collection consolidated CSV / TSV output

XML-C5 builds on the reviewed and validated structural recipe from XML-C4.
It creates one CSV/TSV from at least two reviewed XML sources, with an explicit
`__source_file` and `__source_row` on every output row. Absent fields produce
empty output cells; sources with incompatible column semantics are rejected.
For `separate_table` rules, child tables are exported beside the main CSV with
`__source_file` and the original `__parent_record` linkage.

## CLI (module-local to avoid conflicts with a customized central CLI)

```sh
python -m ml_lab.data.xml_collection_export --recipe analysis/data/xml_collection_recipe.json --preview
python -m ml_lab.data.xml_collection_export --recipe analysis/data/xml_collection_recipe.json --output analysis/data/xml_collection.csv --confirm
```

## Browser

Visit `/data/xml/collection/export` in the local ML_Lab UI, paste the exported C4
recipe JSON, preview the result, choose a destination, and explicitly confirm
output generation. No changes to the current C4 recipe template or central CLI
are required. The browser and CLI share the same validation/extraction API.

## API

- `ml_lab.data.xml_collection_export.preview_xml_collection_export(recipe)` returns
  a bounded, non-persisted preview for human review.
- `ml_lab.data.xml_collection_export.materialize_xml_collection(recipe, output, confirm=True)`
  performs a bounded full extraction and returns the manifest.

The C4 recipe is rechecked against the original source hashes and single-file
record-root, field, and repeated-branch rules before extraction. Materialization
rejects truncated output, changed XML, column-name/meaning conflicts, unsafe
symlinks, and preexisting output unless explicitly overwritten. Outputs are
staged and committed with rollback on publication errors. **Do not treat C5's
`.collection.json` manifest as authoritative provenance**: multi-file lineage,
recovery, and formal provenance integration are XML-C6 work. C7 will address
larger/streaming workloads; current C5 execution uses bounded in-memory tables.
