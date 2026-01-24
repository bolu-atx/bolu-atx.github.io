.PHONY: help serve build build-preview clean install shell clean-cache

.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  serve          Start local dev server with drafts + future posts (localhost:4000)"
	@echo "  build          Production build (no drafts/future)"
	@echo "  build-preview  Build with drafts + future posts"
	@echo "  install        Install Ruby dependencies"
	@echo "  clean          Remove generated _site and cache"
	@echo "  clean-cache    Remove Docker gem cache volume"
	@echo "  shell          Open bash shell in Docker container"

VOLUME_NAME := jekyll-gems-bolu-blog
RUBY_IMAGE := ruby:3.2
TTY_FLAG := $(shell [ -t 0 ] && echo "-it")

DOCKER_RUN := docker run --rm $(TTY_FLAG) \
	--volume="$(PWD):/srv/jekyll" \
	--volume="$(VOLUME_NAME):/usr/local/bundle" \
	-w /srv/jekyll

DOCKER_RUN_WITH_PORT := $(DOCKER_RUN) -p 4000:4000

# Ensure gem cache volume exists
.volume:
	@docker volume create $(VOLUME_NAME) &>/dev/null || true

install: .volume
	$(DOCKER_RUN) $(RUBY_IMAGE) bundle install

serve: .volume
	$(DOCKER_RUN_WITH_PORT) $(RUBY_IMAGE) \
		bash -c "bundle install --quiet && bundle exec jekyll serve --drafts --future --host 0.0.0.0"

build: .volume
	$(DOCKER_RUN) $(RUBY_IMAGE) \
		bash -c "bundle install --quiet && bundle exec jekyll build"

build-preview: .volume
	$(DOCKER_RUN) $(RUBY_IMAGE) \
		bash -c "bundle install --quiet && bundle exec jekyll build --drafts --future"

clean:
	rm -rf _site .jekyll-cache .jekyll-metadata

shell: .volume
	$(DOCKER_RUN_WITH_PORT) $(RUBY_IMAGE) bash

clean-cache:
	docker volume rm $(VOLUME_NAME) 2>/dev/null || true
