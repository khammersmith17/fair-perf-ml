#!/bin/bash


if [[ -n "$VIRTUAL_ENV" ]]; then
	source ~/Documents/fair-ml/.venv/bin/activate
fi

pyflakes ~/Documents/fair-ml/python/fair_ml/*.py

black ~/Documents/fair-ml/python/fair_ml/*.py
