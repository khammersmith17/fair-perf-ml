#!/bin/bash


if [[ -n "$VIRTUAL_ENV"]]; then
	source ~/Documents/fair-ml/.venv/bin/activate
fi


cd ~/Documents/fair-ml
maturin develop
cd tests
python3 tests.py
