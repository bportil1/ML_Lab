# XML-C4 — shared extraction recipe validation

XML-C4 extends XML-C1 inventory, XML-C2 structure comparison and XML-C3 reviewed
membership. It **does not** perform extraction or generate a CSV. Keep the
existing single-file XML inspection/extraction workflow unchanged.

## Browser workflow

1. Visit **Data → XML collection review** at `/data/xml/collection/review`.
2. Inventory a directory, select at least two sources, and acknowledge any
   structural variants. Incompatible source pairs are blocked.
3. Select **Configure shared extraction recipe (C4)**. Choose a canonical record
   root present in every selected source and inspect its candidate fields.
4. Choose field IDs. If some selected fields are absent from particular sources,
   explicitly allow missing fields; the proposal then records per-source gaps.
5. Inspect repeated branches and assign a strategy to each required branch,
   including strategy-specific options where needed (join delimiter, aggregate
   operation, pivot fields, separate-table name).
6. Validate and export the JSON recipe. A blocked recipe cannot be exported.

The UI reruns discovery, structural comparison, and source SHA-256 validation for
submissions. Source XML files are neither altered nor uploaded.

## CLI

The C4 CLI is module-local to avoid conflicts with site-customized central CLI
registrations. The Data Lab browser route is integrated; callers may import the
functions directly from `ml_lab.data.xml_collection_recipe`.


Inspect shared record-root and field options:

```sh
python -m ml_lab.data.xml_collection_recipe ./xml_samples --source a.xml --source b.xml \
  --acknowledge-variants --record-root /rows/item --inspect
```

Validate and save a collection recipe:

```sh
python -m ml_lab.data.xml_collection_recipe ./xml_samples --source a.xml --source b.xml \
  --acknowledge-variants --allow-missing-fields --record-root /rows/item \
  --field text:/rows/item/name --field text:/rows/item/tag \
  --rules-json ./rules.json --output analysis/data/xml_collection_recipe.json
```

`rules.json` is a JSON array of the existing single-file repeated-branch rule
objects, for example:

```json
[{"branch_canonical_path":"/rows/item/tag","strategy":"join","join_delimiter":","}]
```

## Contracts and limitations

- API: `ml_lab.data.xml_collection_recipe.inspect_xml_collection_recipe(...)` returns
  `ml-lab.xml-collection-field-catalog@1`.
- API: `ml_lab.data.xml_collection_recipe.build_xml_collection_recipe(...)` returns
  `ml-lab.xml-collection-recipe@1` with per-source presence, missing fields,
  required branches, rule diagnostics, source hashes and blocking reasons.
- Only `status=ready_for_collection_preview` and
  `structural_recipe_validated=true` permit exporting a recipe JSON.
- `merge_authorized=false` and `csv_generated=false` are always maintained.
- Root membership and field identities are canonical XML paths, not display
  prefixes. Structural similarity does not imply semantic equivalence.
- Value typing, aggregate numeric suitability, pivot collision behavior,
  final column reconciliation, null alignment and consolidated tabular export
  require XML-C5 execution and preview validation. Never treat a C4 recipe as
  evidence that extraction succeeded or that outputs may be merged.
- Input size limits, symlink restrictions and XML parser protections from XML-C1
  remain in effect. Re-check sources at C5 time; a saved C4 JSON is not an
  authorization to operate on changed files.
