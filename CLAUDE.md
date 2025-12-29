# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal tech blog (bolu.dev) built with Jekyll 4.3.3, deployed to GitHub Pages via GitHub Actions.

## Development Commands

```bash
# Install dependencies
bundle install

# Run local dev server (includes drafts and future posts)
bundle exec jekyll serve --drafts --future

# Build for production
bundle exec jekyll build

# Alternative: Docker-based development
./docker_jekyll.sh serve --drafts --future
```

Local server runs at http://localhost:4000

## Content Structure

**Posts**: `_posts/YYYY-MM-DD-slug.md` - Published blog posts

**Drafts**: `_drafts/` - Work-in-progress posts (visible with `--drafts` flag)

**Post front matter format**:
```yaml
---
layout: post
title:  "Post Title"
date:   YYYY-MM-DD HH:MM:SS -0700
tags: tag1 tag2
author: bolu-atx
categories: programming
---
```

Use `<!--more-->` to mark the excerpt separator.

**Tag pages**: `tag/*.md` - Each tag needs a corresponding page file.

## Architecture

- `_layouts/`: default.html → post.html/page.html chain
- `_includes/`: Reusable components (header, footer, analytics, read-time estimator)
- `_sass/`: SCSS partials compiled into site CSS
- `css/`: Main stylesheet entry point
- `assets/`: Images and static files

## Deployment

Push to `master` triggers GitHub Actions workflow (`.github/workflows/jekyll.yml`) which builds and deploys to GitHub Pages.
