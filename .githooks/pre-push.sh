#!/bin/sh
set -e

echo "running tests before push..."
pytest -q

echo "tests passed."