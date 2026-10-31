#!/bin/sh
# Copy results/ (plots, tables, logs) and profiles/ back from gpu32.
set -e
HOST=ss12026@gpu32.doc.ic.ac.uk
SSH="ssh -q -J shell3 -i $HOME/.ssh/doc_ed25519/id_ed25519"
cd "$(dirname "$0")/.."
rsync -az -e "$SSH" "$HOST:flashattn-from-scratch/results/" results/ 2>&1 | grep -v -E "WARNING|prosecuted|monitored|Authorized|College|Use of IT|imperial.ac.uk/dev|DO NOT" || true
rsync -az -e "$SSH" "$HOST:flashattn-from-scratch/profiles/" profiles/ 2>/dev/null || true
