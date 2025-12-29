#!/bin/bash
set -euo pipefail

if ! command -v docker &> /dev/null; then
  echo "Error: Docker not found" >&2
  exit 1
fi

VOLUME_NAME="jekyll-gems-bolu-blog"
RUBY_IMAGE="ruby:3.2"

# Create gem cache volume if it doesn't exist
docker volume create "$VOLUME_NAME" &>/dev/null || true

# Use -it only when running interactively
TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

docker run --rm $TTY_FLAG \
  --volume="$PWD:/srv/jekyll" \
  --volume="$VOLUME_NAME:/usr/local/bundle" \
  -w /srv/jekyll \
  -p 4000:4000 \
  "$RUBY_IMAGE" \
  bash -c "bundle install --quiet && bundle exec jekyll $*"
