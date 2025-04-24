#!/bin/bash

# Check if DYLD_FALLBACK_LIBRARY_PATH is not set
if [ -z "$DYLD_FALLBACK_LIBRARY_PATH" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
fi

# Check if mkdocs is already running on port 8001
if ! lsof -i :8001 > /dev/null 2>&1; then
    mkdocs serve -a localhost:8001
else
    echo "MkDocs server is already running on port 8001"
fi                     