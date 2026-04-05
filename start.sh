#!/bin/bash
# Start the Flask application with Gunicorn
gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 8 --timeout 0