#!/usr/bin/env bash
# Flash the PyroSight data partitions ("model", "audio") without rebuilding
# or reflashing the application.
#
#   tools/flash_partitions.sh -p /dev/ttyUSB0 --model ml/out/pyrosight.espdl
#   tools/flash_partitions.sh -p /dev/ttyUSB0 --audio            # builds clips first
#   tools/flash_partitions.sh -p /dev/ttyUSB0 --audio build/audio/audio.bin
#   tools/flash_partitions.sh -p /dev/ttyUSB0 --erase-model      # force the classical detector
#
# Offsets/sizes come from firmware/partitions.csv. Uses ESP-IDF's parttool.py
# when $IDF_PATH is set, otherwise esptool (python3 -m esptool) directly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV="$ROOT/firmware/partitions.csv"
PORT="${ESPPORT:-}"
BAUD="${ESPBAUD:-921600}"
MODEL=""
AUDIO=""
WANT_AUDIO=0
ERASE_MODEL=0

usage() { sed -n '2,13p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--port) PORT="$2"; shift 2 ;;
    -b|--baud) BAUD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --audio)
      WANT_AUDIO=1
      if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then AUDIO="$2"; shift; fi
      shift ;;
    --erase-model) ERASE_MODEL=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[ -n "$PORT" ] || { echo "serial port required (-p or \$ESPPORT)" >&2; exit 1; }
[ -n "$MODEL" ] || [ "$WANT_AUDIO" = 1 ] || [ "$ERASE_MODEL" = 1 ] || usage 1

# Partition offset and size (decimal) from the CSV.
part() {
  awk -F, -v name="$1" '
    /^[[:space:]]*#/ { next }
    { gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $4); gsub(/[[:space:]]/, "", $5) }
    $1 == name { print $4, $5; found = 1 }
    END { if (!found) exit 1 }' "$CSV"
}
to_dec() { printf '%d' "$1"; }
size_of_file() { wc -c < "$1" | tr -d ' '; }

esptool() {
  if command -v esptool.py >/dev/null 2>&1; then esptool.py "$@"; else python3 -m esptool "$@"; fi
}

write_part() {  # name file
  local name="$1" file="$2" off size
  read -r off size < <(part "$name") || { echo "partition $name not in $CSV" >&2; exit 1; }
  local fsize; fsize=$(size_of_file "$file")
  if [ "$fsize" -gt "$(to_dec "$size")" ]; then
    echo "$file ($fsize B) does not fit partition $name ($size)" >&2; exit 1
  fi
  echo "writing $file ($fsize B) to $name @ $off"
  if [ -n "${IDF_PATH:-}" ] && [ -f "$IDF_PATH/components/partition_table/parttool.py" ]; then
    python3 "$IDF_PATH/components/partition_table/parttool.py" --port "$PORT" --baud "$BAUD" \
      write_partition --partition-name "$name" --input "$file"
  else
    esptool --chip esp32p4 --port "$PORT" --baud "$BAUD" write_flash "$off" "$file"
  fi
}

erase_part() {
  local name="$1" off size
  read -r off size < <(part "$name")
  echo "erasing $name @ $off ($size)"
  esptool --chip esp32p4 --port "$PORT" --baud "$BAUD" erase_region "$off" "$size"
}

if [ "$ERASE_MODEL" = 1 ]; then erase_part model; fi

if [ -n "$MODEL" ]; then
  [ -f "$MODEL" ] || { echo "no such model file: $MODEL" >&2; exit 1; }
  write_part model "$MODEL"
fi

if [ "$WANT_AUDIO" = 1 ]; then
  if [ -z "$AUDIO" ]; then
    read -r _ asize < <(part audio)
    python3 "$ROOT/tools/make_audio_clips.py" --out "$ROOT/build/audio" --partition-size "$asize"
    AUDIO="$ROOT/build/audio/audio.bin"
  fi
  [ -f "$AUDIO" ] || { echo "no such audio image: $AUDIO" >&2; exit 1; }
  write_part audio "$AUDIO"
fi
echo done
