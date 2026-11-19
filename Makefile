PY ?= .venv/bin/python
SPEC_DIR ?= $(HOME)/spec/cpu2026
export SPEC_DIR

.PHONY: all ramulator venv tracer spec-install spec-build spec-trace calibrate ai spec spec-smoke patterns figures report test

all: ai spec patterns figures report

ramulator:
	cd ramulator2 && ./build.sh

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

tracer:
	$(MAKE) -C tracer

# Install SPEC CPU 2026 from the ISO in this directory (needs ~10 GB).
spec-install:
	mkdir -p $(HOME)/spec/iso && bsdtar -xf cpu2026-1.0.1.iso -C $(HOME)/spec/iso \
	  --exclude redistributable_sources --exclude PTDaemon --exclude Docs && chmod -R u+w $(HOME)/spec/iso
	mkdir -p spec/.compat-lib && [ -e /usr/lib/libcrypt.so.1 ] || ln -sf /usr/lib/libcrypt.so.2 spec/.compat-lib/libcrypt.so.1
	cd $(HOME)/spec/iso && LD_LIBRARY_PATH=$(CURDIR)/spec/.compat-lib ./install.sh -d $(SPEC_DIR) -f -u linux-x86_64

spec-build: tracer
	REBUILD=1 spec/build_spec.sh refrate

spec-trace:
	python3 spec/trace_spec.py --skip 30000000000 --insts 200000000 --jobs 8

calibrate:
	$(PY) -m memarch calibrate $(ARGS)

ai:
	$(PY) -m memarch ai $(ARGS)

spec:
	$(PY) -m memarch spec --jobs 10 $(ARGS)

# Extra CLI flags for calibrate/ai/spec/patterns, e.g. make spec ARGS="--archs gpu_hbm3 --merge"
ARGS ?=

# One short replay into RAM (/dev/shm) to check the toolchain; results/ is untouched.
SMOKE_OUT ?= /dev/shm/memarch-smoke
spec-smoke:
	$(PY) -m memarch spec --archs gpu_hbm3 --benchmarks 708.sqlite_r --cores 1 --insts 2000000 --jobs 1 --out $(SMOKE_OUT)

# Access-pattern characterization of the SPEC traces: locality + inter-request PSD at the DRAM controllers
patterns:
	$(PY) -m memarch patterns --jobs 6 $(ARGS)

figures:
	$(PY) -m memarch figures

report:
	$(PY) report/build_report.py

test:
	$(PY) -m pytest -q
