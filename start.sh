#!/usr/bin/env bash

pip install --upgrade pip setuptools wheel

gunicorn app:app
