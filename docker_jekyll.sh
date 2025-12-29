#!/bin/bash

if ! command -v docker &> /dev/null; then
  echo "Docker not found. Cannot continue."
  exit 1
fi

# Use -it only when running interactively (terminal attached)
TTY_FLAG=""
if [ -t 0 ]; then
  TTY_FLAG="-it"
fi

docker run --rm $TTY_FLAG \
  --volume="$PWD:/srv/jekyll" \
  -w /srv/jekyll \
  -p 4000:4000 \
  ruby:3.2 \
  bash -c "bundle install && bundle exec jekyll $*"
