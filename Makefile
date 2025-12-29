.PHONY: serve build clean install shell clean-cache

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

clean:
	rm -rf _site .jekyll-cache .jekyll-metadata

shell: .volume
	$(DOCKER_RUN_WITH_PORT) $(RUBY_IMAGE) bash

clean-cache:
	docker volume rm $(VOLUME_NAME) 2>/dev/null || true
