.PHONY: run clean

run:
	python scripts/run_pipeline.py

clean:
	rm -rf artifacts
