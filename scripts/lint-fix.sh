#!/bin/sh -e
set -ex

echo "Fixing code..."
.venv/bin/ruff check src tests scripts --fix

echo
echo "Formatting code..."
.venv/bin/ruff format src tests scripts

echo
echo "Done"
