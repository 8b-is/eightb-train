.PHONY: dry sft seal
dry:
	uv run train.py --config config/base-27b.yaml --dry-run
	uv run pqseal.py --weights x --out /tmp/deck --dry-run
sft:
	uv run train.py --config config/base-27b.yaml
seal:
	uv run pqseal.py --weights $${MERGED_WEIGHTS:?set MERGED_WEIGHTS} --out deck/
