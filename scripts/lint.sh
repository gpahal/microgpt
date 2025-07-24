#!/bin/sh -e
set -ex

echo "Checking types..."
.venv/bin/mypy src tests scripts

echo
echo "Checking code..."
.venv/bin/ruff check src tests scripts

echo
echo "Checking formatting..."
.venv/bin/ruff format src tests scripts --check

echo
echo "Done"
