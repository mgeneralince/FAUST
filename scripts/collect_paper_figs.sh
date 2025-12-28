#!/usr/bin/env bash
set -euo pipefail

FLS=${1:-notes/notes.fls}
DEST=${2:-notes/figs/paper}
MODE=${3:-symlink}   

mkdir -p "$DEST"

BASE="$(awk '/^PWD /{print $2; exit}' "$FLS")"
if [[ -z "$BASE" ]]; then
  BASE="$(cd "$(dirname "$FLS")" && pwd)"
fi

count=0

grep '^INPUT ./figs/' "$FLS" | awk '{print $2}' | sort -u | while IFS= read -r p; do
  rel="${p#./}"           
  src="$BASE/$rel"         
  out_rel="${rel#figs/}"   
  dst="$DEST/$out_rel"     

  [[ -f "$src" ]] || continue
  mkdir -p "$(dirname "$dst")"

  if [[ "$MODE" == "copy" ]]; then
    cp -f "$src" "$dst"
  else
    ln -sf "$src" "$dst"
  fi

  count=$((count+1))
done

echo "Collected figures into: $DEST (mode=$MODE)"
echo "Note: macOS pipes run 'while' in a subshell, so count may print as 0 even if files were created."
