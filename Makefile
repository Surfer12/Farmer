SHELL := /bin/bash

.PHONY: help test-qualia start-http start-https setup-https setup-https-prod clean-certs cfd-run cfd-plot

help:
	@echo "Available targets:"
	@echo "  test-qualia   - Run qualia tests"
	@echo "  start-http    - Start assistants with HTTP"
	@echo "  start-https   - Start assistants with HTTPS"
	@echo "  setup-https   - Set up HTTPS certificates (dev)"
	@echo "  setup-https-prod - Set up HTTPS certificates (production)"
	@echo "  clean-certs   - Remove SSL certificates"

test-qualia:
	mvn clean test

start-http:
	python3 assistants/launcher/launch_all.py

start-https:
	./scripts/start_https.sh

setup-https:
	python3 scripts/setup_https.py --mode dev

setup-https-prod:
	@echo "Enter domain name:" && read domain && \
	echo "Enter email for Let's Encrypt:" && read email && \
	python3 scripts/setup_https.py --mode prod --domain $$domain --email $$email

clean-certs:
	rm -rf certs/

cfd-run:
	python3 /workspace/cfd_mojo/solver_python.py --nx 300 --ny 150 --u 2.0 --aoa 10.0 --nu 1e-6 --steps 1200 --save_every 300 --out /workspace/cfd_mojo/out --chord 0.28 --thickness 0.06 --camber 0.02 --fin_angle 6.5 | cat

cfd-plot:
	MPLBACKEND=Agg python3 /workspace/cfd_mojo/plot_results.py --in_dir /workspace/cfd_mojo/out --out_png /workspace/cfd_mojo/pressure_map.png | cat


