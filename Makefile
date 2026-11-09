.PHONY: dry sft merge seal e2e
dry:
	uv run train.py --config config/base-27b.yaml --dry-run
	uv run merge_adapter.py --base x --adapter y --out /tmp/m --dry-run
	uv run pqseal.py --weights x --out /tmp/deck --dry-run
sft:
	uv run train.py --config config/base-27b.yaml
merge:
	uv run merge_adapter.py --base $${BASE:?set BASE} --adapter $${ADAPTER:?set ADAPTER} --out $${OUT:-/workspace/out/eightb-27b/merged.safetensors}
seal:
	uv run pqseal.py --weights $${MERGED_WEIGHTS:?set MERGED_WEIGHTS} --out deck/
e2e:
	bash scripts/run_a6000.sh config/base-27b.yaml
