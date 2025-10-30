#!/bin/bash
set -e
echo "Starting Medical AI Model Server..."
exec uvicorn deploy.serve_model:app --host 0.0.0.0 --port 8000
