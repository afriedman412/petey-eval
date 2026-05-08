VENV = venv
SYSTEM_PYTHON ?= /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

PROJECT = petey-dev
REGION = us-east1
IMAGE = us-east1-docker.pkg.dev/$(PROJECT)/petey/benchmark:latest
JOB_NAME = petey-benchmark

.PHONY: venv install clean build deploy run dry-run

venv:
	arch -arm64 $(SYSTEM_PYTHON) -m venv $(VENV)

install: venv
	arch -arm64 $(PIP) install -r requirements.txt

# ---------- Cloud Run Job ----------

build:
	gcloud config set project $(PROJECT)
	docker build --platform linux/amd64 -t $(IMAGE) .
	docker push $(IMAGE)

deploy: build
	gcloud config set project $(PROJECT)
	gcloud run jobs create $(JOB_NAME) \
		--image=$(IMAGE) \
		--region=$(REGION) \
		--memory=4Gi \
		--cpu=4 \
		--task-timeout=86400 \
		--max-retries=0 \
		--set-env-vars="$$(cat ../.env | grep -v '^#' | grep -v 'GOOGLE_APPLICATION_CREDENTIALS' | grep '=' | tr '\n' ',')" \
		--args="--gcs,--runs,3,--datasets,medical,par_simple,par_detailed" \
		2>/dev/null || \
	gcloud run jobs update $(JOB_NAME) \
		--image=$(IMAGE) \
		--region=$(REGION) \
		--memory=4Gi \
		--cpu=4 \
		--task-timeout=86400 \
		--max-retries=0 \
		--set-env-vars="$$(cat ../.env | grep -v '^#' | grep -v 'GOOGLE_APPLICATION_CREDENTIALS' | grep '=' | tr '\n' ',')" \
		--args="--gcs,--runs,3,--datasets,medical,par_simple,par_detailed"

# Run with custom args: make run ARGS="--models gpt-4.1 --datasets medical --runs 1"
run:
	gcloud config set project $(PROJECT)
	gcloud run jobs execute $(JOB_NAME) \
		--region=$(REGION) \
		$(if $(ARGS),--args="$(ARGS)",) \
		--wait

dry-run:
	$(PYTHON) benchmark.py --dry-run

clean:
	rm -rf $(VENV) *.egg-info results/ .pdf_cache/
