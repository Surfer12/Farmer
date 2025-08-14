SHELL := /bin/bash

.PHONY: help test-qualia start-http start-https setup-https setup-https-prod clean-certs

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


