#!/bin/sh
# Sync the repo to gpu32 (via shell3) and run a command there inside the repo.
# Usage: scripts/remote.sh "make && python tests/test_gemm.py build/01_naive"
# Needs an ssh-agent holding the DoC key (SSH_AUTH_SOCK).
set -e
HOST=ss12026@gpu32.doc.ic.ac.uk
SSH="ssh -q -J shell3 -i $HOME/.ssh/doc_ed25519/id_ed25519"
REMOTE_DIR=flashattn-from-scratch
cd "$(dirname "$0")/.."
rsync -az --exclude build --exclude .git --exclude results/__pycache__ -e "$SSH" ./ "$HOST:$REMOTE_DIR/"
$SSH "$HOST" "export PATH=/vol/cuda/13.1.0/bin:\$PATH LD_LIBRARY_PATH=/vol/cuda/13.1.0/lib64:\$LD_LIBRARY_PATH; \
  . /vol/bitbucket/ss12026/fa-venv/bin/activate 2>/dev/null || true; cd $REMOTE_DIR && $1" 2>&1 \
  | grep -v -E "WARNING|prosecuted|may be monitored|Authorized usage|College Terms|Use of IT Resources|imperial.ac.uk/dev-sites|DO NOT RUN"
