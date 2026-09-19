#!/usr/bin/env bash
# Unlock the API key once for this login session.
#
# Decrypting the `pass` entry needs a YubiKey touch, and the touch window is
# short, so this script does it exactly once and caches the result in
# $XDG_RUNTIME_DIR -- tmpfs, so RAM-backed and gone at logout. Nothing is
# written to disk and nothing is printed.
#
# Usage:
#   scripts/prime-key.sh          # fetch and cache (prompts for a touch)
#   scripts/prime-key.sh --check  # report whether the cache is warm
#   scripts/prime-key.sh --clear  # drop the cached key
#
# To get the key into a shell's environment:
#   export TYPESAFE_API_KEY="$(scripts/prime-key.sh --print)"

set -euo pipefail

PASS_ENTRY="typesafe.ai/perf"
CACHE="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/jev-eval-api-key"

case "${1:-}" in
  --check)
    if [[ -s "$CACHE" ]]; then
      echo "session key cache: warm ($CACHE)"
    else
      echo "session key cache: cold -- run scripts/prime-key.sh"
      exit 1
    fi
    exit 0
    ;;
  --clear)
    rm -f "$CACHE"
    echo "session key cache cleared"
    exit 0
    ;;
esac

if [[ ! -s "$CACHE" ]]; then
  echo "Fetching API key via \`pass $PASS_ENTRY\` -- touch your YubiKey when it blinks." >&2
  umask 077
  pass "$PASS_ENTRY" | head -n1 | tr -d '\n' > "$CACHE"
  if [[ ! -s "$CACHE" ]]; then
    rm -f "$CACHE"
    echo "failed to retrieve key" >&2
    exit 1
  fi
  echo "key cached for this session in \$XDG_RUNTIME_DIR (tmpfs, not disk)" >&2
fi

if [[ "${1:-}" == "--print" ]]; then
  cat "$CACHE"
fi
