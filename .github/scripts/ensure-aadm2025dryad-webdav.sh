#!/usr/bin/env bash
set -euo pipefail

mask_secret() {
  if [ -n "${1:-}" ]; then
    echo "::add-mask::$1"
  fi
}

mask_secret "${AADM2025DRYAD_DATA_WEBDAV_URL:-}"
mask_secret "${AADM2025DRYAD_DATA_KEY:-}"
mask_secret "${AADM2025DRYAD_WEBDAV_CRED:-}"

cache_root="${HOME}/github-runner/.cache/datasets"
cache_dataset_root="${cache_root}/AADM2025Dryad"
mkdir -p "${cache_root}"

find_rf_root() {
  for root in "$@"; do
    if [ -z "${root:-}" ] || [ ! -d "${root}" ]; then
      continue
    fi
    case "$(basename "${root}")" in
      "RF Sensor and Radar"|"RF_Sensor_and_Radar")
        printf '%s\n' "${root}"
        return 0
        ;;
    esac
    found="$(find "${root}" -maxdepth 8 -type d \( -name "RF Sensor and Radar" -o -name "RF_Sensor_and_Radar" \) -print -quit 2>/dev/null || true)"
    if [ -n "${found}" ]; then
      printf '%s\n' "${found}"
      return 0
    fi
  done
  return 1
}

resolve_dataset() {
  rf_root="$(find_rf_root \
    "${DATASET_ROOT:-}" \
    "${cache_dataset_root}" \
    "${cache_root}" \
    "${HOME}/.cache/datasets/AADM2025Dryad" \
    "${HOME}/.cache/datasets" \
    "/srv/datasets/AADM2025Dryad" \
    "/srv/datasets" \
    "/data/AADM2025Dryad" \
    "/data" \
    "/mnt/data/AADM2025Dryad" \
    "/mnt/data" \
    "${HOME}/AADM2025Dryad" \
    "${HOME}" \
    "/home/github-runner/AADM2025Dryad" \
    "/home/github-runner" || true)"
  if [ -z "${rf_root}" ]; then
    return 1
  fi
  resolved_dataset_root="$(dirname "${rf_root}")"
  printf '%s\n' "${resolved_dataset_root}" > "${RUNNER_TEMP}/resolved_aadm2025dryad_root.txt"
  printf '%s\n' "${rf_root}" > "${RUNNER_TEMP}/resolved_aadm2025dryad_rf_root.txt"
  return 0
}

ensure_rclone() {
  if command -v rclone >/dev/null 2>&1; then
    return 0
  fi
  echo "rclone is not installed; installing a temporary copy in RUNNER_TEMP."
  mkdir -p "${RUNNER_TEMP}/rclone-bin" "${RUNNER_TEMP}/rclone-download"
  curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o "${RUNNER_TEMP}/rclone.zip"
  python -m zipfile -e "${RUNNER_TEMP}/rclone.zip" "${RUNNER_TEMP}/rclone-download"
  rclone_bin="$(find "${RUNNER_TEMP}/rclone-download" -type f -name rclone -print -quit)"
  if [ -z "${rclone_bin}" ]; then
    echo "Could not find rclone binary after extracting download." >&2
    exit 1
  fi
  cp "${rclone_bin}" "${RUNNER_TEMP}/rclone-bin/rclone"
  chmod +x "${RUNNER_TEMP}/rclone-bin/rclone"
  export PATH="${RUNNER_TEMP}/rclone-bin:${PATH}"
  echo "${RUNNER_TEMP}/rclone-bin" >> "${GITHUB_PATH}"
}

extract_zip_archive() {
  local archive="$1"
  local target="$2"

  python - "${archive}" "${target}" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
target = Path(sys.argv[2])

if not zipfile.is_zipfile(archive):
    raise SystemExit(2)

with zipfile.ZipFile(archive) as zip_file:
    zip_file.extractall(target)
PY
}

download_webdav_archive() {
  local staging="$1"
  local archive="${RUNNER_TEMP}/AADM2025Dryad.webdav.zip"
  local base_url="${AADM2025DRYAD_DATA_WEBDAV_URL%/}"
  local urls=("${AADM2025DRYAD_DATA_WEBDAV_URL}")
  local origin=""
  local token=""

  add_url_variant() {
    local candidate="$1"
    local existing=""
    for existing in "${urls[@]}"; do
      if [ "${existing}" = "${candidate}" ]; then
        return 0
      fi
    done
    urls+=("${candidate}")
  }

  case "${base_url}" in
    */download) ;;
    *) add_url_variant "${base_url}/download" ;;
  esac

  case "${base_url}" in
    */public.php/webdav*)
      origin="${base_url%%/public.php/webdav*}"
      add_url_variant "${origin}/index.php/s/${AADM2025DRYAD_DATA_KEY}/download"
      add_url_variant "${origin}/s/${AADM2025DRYAD_DATA_KEY}/download"
      ;;
    */remote.php/dav/public-files/*)
      origin="${base_url%%/remote.php/dav/public-files/*}"
      token="${base_url#*/remote.php/dav/public-files/}"
      token="${token%%/*}"
      add_url_variant "${origin}/index.php/s/${token}/download"
      add_url_variant "${origin}/s/${token}/download"
      ;;
  esac

  for url in "${urls[@]}"; do
    rm -f "${archive}"
    echo "Trying authenticated WebDAV archive download fallback."
    if ! curl --fail --location --retry 5 --retry-delay 20 --connect-timeout 60 \
      --user "${AADM2025DRYAD_DATA_KEY}:${AADM2025DRYAD_WEBDAV_CRED}" \
      --output "${archive}" \
      "${url}"; then
      continue
    fi

    du -sh "${archive}" || true
    if extract_zip_archive "${archive}" "${staging}"; then
      return 0
    fi

    echo "Downloaded WebDAV response was not a ZIP archive; trying next URL variant." >&2
  done

  return 1
}

download_dataset() {
  if [ -z "${AADM2025DRYAD_DATA_WEBDAV_URL:-}" ] || [ -z "${AADM2025DRYAD_DATA_KEY:-}" ] || [ -z "${AADM2025DRYAD_WEBDAV_CRED:-}" ]; then
    echo "AADM2025Dryad was not found locally, and the WebDAV source secrets are incomplete." >&2
    echo "Required secrets: AADM2025DRYAD_DATA_WEBDAV_URL, AADM2025DRYAD_DATA_KEY, AADM2025DRYAD_DATA_PASSWORD." >&2
    exit 1
  fi

  ensure_rclone
  staging="${cache_dataset_root}.tmp.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.${TEST_FLIGHT:-job}"
  rm -rf "${staging}"
  mkdir -p "${staging}"

  obscured_cred="$(rclone obscure "${AADM2025DRYAD_WEBDAV_CRED}")"
  echo "Dataset not found locally; downloading WebDAV source into ${cache_dataset_root}."
  if ! rclone copy ":webdav:" "${staging}" \
    --webdav-url "${AADM2025DRYAD_DATA_WEBDAV_URL}" \
    --webdav-vendor owncloud \
    --webdav-user "${AADM2025DRYAD_DATA_KEY}" \
    --webdav-pass "${obscured_cred}" \
    --progress \
    --transfers 8 \
    --checkers 16; then
    echo "rclone WebDAV copy failed; trying authenticated archive download fallback." >&2
    rm -rf "${staging}"
    mkdir -p "${staging}"
    if ! download_webdav_archive "${staging}"; then
      echo "WebDAV archive fallback failed." >&2
      rm -rf "${staging}"
      exit 1
    fi
  fi

  if ! find_rf_root "${staging}" >/dev/null; then
    echo "WebDAV download completed, but no RF Sensor and Radar directory was found in ${staging}." >&2
    find "${staging}" -maxdepth 4 -type d | sed -n '1,160p' >&2 || true
    rm -rf "${staging}"
    exit 1
  fi

  rm -rf "${cache_dataset_root}"
  mv "${staging}" "${cache_dataset_root}"
  echo "Cached dataset at ${cache_dataset_root}"
}

if ! resolve_dataset; then
  echo "AADM2025Dryad local cache miss. Cache root: ${cache_root}"
  lock_file="${cache_root}/.AADM2025Dryad.lock"
  (
    flock 9
    if ! resolve_dataset; then
      download_dataset
    fi
  ) 9>"${lock_file}"
fi

if ! resolve_dataset; then
  echo "Could not resolve AADM2025Dryad after cache/download step." >&2
  exit 1
fi

resolved_dataset_root="$(cat "${RUNNER_TEMP}/resolved_aadm2025dryad_root.txt")"
resolved_rf_root="$(cat "${RUNNER_TEMP}/resolved_aadm2025dryad_rf_root.txt")"

{
  echo "DATASET_ROOT=${resolved_dataset_root}"
  echo "AADM2025DRYAD_DATASET_ROOT=${resolved_dataset_root}"
  echo "RF_SENSOR_AND_RADAR_ROOT=${resolved_rf_root}"
  echo "PERSISTENT_DATASET_CACHE=${cache_dataset_root}"
} >> "${GITHUB_ENV}"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "dataset-root=${resolved_dataset_root}"
    echo "rf-sensor-and-radar-root=${resolved_rf_root}"
    echo "cache-dataset-root=${cache_dataset_root}"
  } >> "${GITHUB_OUTPUT}"
fi

echo "Resolved DATASET_ROOT=${resolved_dataset_root}"
echo "Resolved RF_SENSOR_AND_RADAR_ROOT=${resolved_rf_root}"
du -sh "${resolved_dataset_root}" || true
