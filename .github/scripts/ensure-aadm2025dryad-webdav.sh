#!/usr/bin/env bash
set -euo pipefail

AADM2025DRYAD_DATA_URL="${AADM2025DRYAD_DATA_URL:-${AADM2025DRYAD_URL:-}}"
AADM2025DRYAD_WEBDAV_CRED="${AADM2025DRYAD_WEBDAV_CRED:-${AADM2025DRYAD_DATA_PASSWORD:-}}"

mask_secret() {
  if [ -n "${1:-}" ]; then
    echo "::add-mask::$1"
  fi
}

mask_secret "${AADM2025DRYAD_DATA_URL:-}"
mask_secret "${AADM2025DRYAD_DATA_WEBDAV_URL:-}"
mask_secret "${AADM2025DRYAD_DATA_KEY:-}"
mask_secret "${AADM2025DRYAD_WEBDAV_CRED:-}"

cache_root="${HOME}/github-runner/.cache/datasets"
cache_dataset_root="${cache_root}/AADM2025Dryad"
mkdir -p "${cache_root}"

find_rf_root_impl() {
  local allow_transient="$1"
  shift

  for root in "$@"; do
    if [ -z "${root:-}" ] || [ ! -d "${root}" ]; then
      continue
    fi
    if [ "${allow_transient}" != "true" ]; then
      case "${root}" in
        *AADM2025Dryad.tmp.*)
          continue
          ;;
      esac
    fi
    case "$(basename "${root}")" in
      "RF Sensor and Radar"|"RF_Sensor_and_Radar")
        printf '%s\n' "${root}"
        return 0
        ;;
    esac
    if [ "${allow_transient}" = "true" ]; then
      found="$(find "${root}" -maxdepth 8 -type d \( -name "RF Sensor and Radar" -o -name "RF_Sensor_and_Radar" \) -print -quit 2>/dev/null || true)"
    else
      found="$(find "${root}" -maxdepth 8 \
        \( -type d -name "AADM2025Dryad.tmp.*" -prune \) -o \
        \( -type d \( -name "RF Sensor and Radar" -o -name "RF_Sensor_and_Radar" \) -print -quit \) \
        2>/dev/null || true)"
    fi
    if [ -n "${found}" ]; then
      printf '%s\n' "${found}"
      return 0
    fi
  done
  return 1
}

find_rf_root() {
  find_rf_root_impl false "$@"
}

find_rf_root_allow_transient() {
  find_rf_root_impl true "$@"
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
  local existing_rclone=""
  local version_output=""

  existing_rclone="$(command -v rclone || true)"
  if [ -n "${existing_rclone}" ]; then
    case "${existing_rclone}" in
      /snap/bin/*|*/snap/bin/*)
        echo "Ignoring Snap-installed rclone at ${existing_rclone}; runner services cannot reliably execute Snap apps."
        ;;
      *)
        if version_output="$("${existing_rclone}" version 2>&1)"; then
          if printf '%s\n' "${version_output}" | grep -q "go/tags: snap"; then
            echo "Ignoring Snap-built rclone at ${existing_rclone}; runner services cannot reliably execute Snap apps."
          else
            return 0
          fi
        else
          echo "Existing rclone at ${existing_rclone} is not usable; installing a temporary copy in RUNNER_TEMP."
        fi
        ;;
    esac
  else
    echo "rclone is not installed; installing a temporary copy in RUNNER_TEMP."
  fi

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
  echo "Using temporary rclone at ${RUNNER_TEMP}/rclone-bin/rclone."
  rclone version | sed -n '1,5p'
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

download_archive_url() {
  local url="$1"
  local archive="$2"
  local use_basic_auth="${3:-false}"
  local auth_args=()
  local curl_meta="${RUNNER_TEMP}/aadm2025dryad_curl_meta.txt"
  local http_code=""
  local content_type=""
  local size_download=""

  if [ "${use_basic_auth}" = "true" ] && [ -n "${AADM2025DRYAD_DATA_KEY:-}" ] && [ -n "${AADM2025DRYAD_WEBDAV_CRED:-}" ]; then
    auth_args=(--user "${AADM2025DRYAD_DATA_KEY}:${AADM2025DRYAD_WEBDAV_CRED}")
  fi

  rm -f "${archive}" "${curl_meta}"
  if ! curl --fail --location --retry 5 --retry-delay 20 --connect-timeout 60 \
    "${auth_args[@]}" \
    --write-out '%{http_code}\t%{content_type}\t%{size_download}\n' \
    --output "${archive}" \
    "${url}" > "${curl_meta}"; then
    return 1
  fi

  IFS=$'\t' read -r http_code content_type size_download < "${curl_meta}" || true
  echo "Downloaded archive candidate: HTTP ${http_code:-unknown}, content-type ${content_type:-unknown}, size ${size_download:-unknown} bytes."
  du -sh "${archive}" || true
  return 0
}

derive_webdav_pairs() {
  python - "${AADM2025DRYAD_DATA_WEBDAV_URL:-}" "${AADM2025DRYAD_DATA_KEY:-}" <<'PY'
from __future__ import annotations

import sys
from urllib.parse import urlsplit, urlunsplit

raw_url = sys.argv[1].strip()
configured_user = sys.argv[2].strip()
if not raw_url:
    raise SystemExit(0)

parts = urlsplit(raw_url)
path = parts.path.rstrip("/")
segments = [segment for segment in path.split("/") if segment]
token = ""
base_segments: list[str] = []

for marker in (("s",), ("index.php", "s")):
    marker_len = len(marker)
    for index in range(0, max(len(segments) - marker_len, 0)):
        if tuple(segments[index:index + marker_len]) == marker and index + marker_len < len(segments):
            token = segments[index + marker_len]
            base_segments = segments[:index]
            break
    if token:
        break

if not token and len(segments) >= 4 and segments[:3] == ["remote.php", "dav", "public-files"]:
    token = segments[3]
    base_segments = []
elif not token:
    for index in range(0, max(len(segments) - 3, 0)):
        if segments[index:index + 3] == ["remote.php", "dav", "public-files"] and index + 3 < len(segments):
            token = segments[index + 3]
            base_segments = segments[:index]
            break

base_path = "/" + "/".join(base_segments) if base_segments else ""
base_url = urlunsplit((parts.scheme, parts.netloc, base_path, "", "")).rstrip("/")

pairs: list[tuple[str, str]] = []

def emit(url: str, user: str) -> None:
    if not url or not user:
        return
    pair = (url, user)
    if pair not in pairs:
        pairs.append(pair)

emit(raw_url, configured_user)
if base_url and token:
    emit(f"{base_url}/public.php/webdav/", token)
    emit(f"{base_url}/public.php/webdav", token)
    emit(f"{base_url}/public.php/dav/files/{token}/", "anonymous")
    emit(f"{base_url}/public.php/dav/files/{token}", "anonymous")
    emit(f"{base_url}/public.php/dav/files/{token}/", token)
    emit(f"{base_url}/public.php/dav/files/{token}", token)
    emit(f"{base_url}/remote.php/dav/public-files/{token}/", token)
    emit(f"{base_url}/remote.php/dav/public-files/{token}", token)
    emit(f"{base_url}/public.php/webdav/", configured_user)
    emit(f"{base_url}/public.php/webdav", configured_user)
    emit(f"{base_url}/public.php/dav/files/{token}/", configured_user)
    emit(f"{base_url}/public.php/dav/files/{token}", configured_user)
    emit(f"{base_url}/remote.php/dav/public-files/{token}/", configured_user)
    emit(f"{base_url}/remote.php/dav/public-files/{token}", configured_user)

for url, user in pairs:
    print(f"{url}\t{user}")
PY
}

copy_webdav_to_staging() {
  local staging="$1"
  local pairs=()
  local pair=""
  local url=""
  local user=""
  local vendor=""
  local vendors=(owncloud nextcloud)
  local variant=0
  local total_variants=0
  local obscured_cred=""
  local probe_output="${RUNNER_TEMP}/aadm2025dryad_webdav_lsf.txt"

  if [ -z "${AADM2025DRYAD_DATA_WEBDAV_URL:-}" ] || [ -z "${AADM2025DRYAD_WEBDAV_CRED:-}" ]; then
    return 1
  fi

  mapfile -t pairs < <(derive_webdav_pairs)
  if [ "${#pairs[@]}" -eq 0 ]; then
    return 1
  fi

  total_variants=$((${#pairs[@]} * ${#vendors[@]}))
  for pair in "${pairs[@]}"; do
    url="${pair%%$'\t'*}"
    user="${pair#*$'\t'}"
    if [ -z "${url}" ] || [ -z "${user}" ]; then
      continue
    fi

    for vendor in "${vendors[@]}"; do
      variant=$((variant + 1))
      rm -rf "${staging:?}"/*
      obscured_cred="$(rclone obscure "${AADM2025DRYAD_WEBDAV_CRED}")"
      echo "Trying WebDAV source variant ${variant}/${total_variants}."
      if ! timeout 45s rclone lsf ":webdav:" \
        --webdav-url "${url}" \
        --webdav-vendor "${vendor}" \
        --webdav-user "${user}" \
        --webdav-pass "${obscured_cred}" \
        --max-depth 2 \
        --contimeout 20s \
        --timeout 30s \
        --low-level-retries 1 \
        --retries 1 \
        > "${probe_output}"; then
        echo "WebDAV source variant ${variant}/${total_variants} could not list the remote; trying next variant." >&2
        continue
      fi
      echo "WebDAV source variant ${variant}/${total_variants} listed successfully; first entries:"
      sed -n '1,20p' "${probe_output}"
      if rclone copy ":webdav:" "${staging}" \
        --webdav-url "${url}" \
        --webdav-vendor "${vendor}" \
        --webdav-user "${user}" \
        --webdav-pass "${obscured_cred}" \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --contimeout 30s \
        --timeout 2m \
        --low-level-retries 2 \
        --retries 2 \
        --retries-sleep 5s; then
        return 0
      fi
    done
  done

  return 1
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
    echo "Trying authenticated WebDAV archive download fallback."
    if ! download_archive_url "${url}" "${archive}" true; then
      continue
    fi

    if extract_zip_archive "${archive}" "${staging}"; then
      return 0
    fi

    echo "Downloaded WebDAV response was not a ZIP archive; trying next URL variant." >&2
  done

  return 1
}

download_direct_archive() {
  local staging="$1"
  local archive="${RUNNER_TEMP}/AADM2025Dryad.zip"

  if [ -z "${AADM2025DRYAD_DATA_URL:-}" ]; then
    return 1
  fi

  echo "Trying direct AADM2025Dryad archive download."
  if download_archive_url "${AADM2025DRYAD_DATA_URL}" "${archive}" && extract_zip_archive "${archive}" "${staging}"; then
    return 0
  fi

  if [ -n "${AADM2025DRYAD_DATA_KEY:-}" ] && [ -n "${AADM2025DRYAD_WEBDAV_CRED:-}" ]; then
    echo "Trying direct AADM2025Dryad archive download with configured credentials."
    if download_archive_url "${AADM2025DRYAD_DATA_URL}" "${archive}" true && extract_zip_archive "${archive}" "${staging}"; then
      return 0
    fi
  fi

  return 1
}

has_webdav_source() {
  [ -n "${AADM2025DRYAD_DATA_WEBDAV_URL:-}" ] && [ -n "${AADM2025DRYAD_DATA_KEY:-}" ] && [ -n "${AADM2025DRYAD_WEBDAV_CRED:-}" ]
}

download_dataset() {
  if [ -z "${AADM2025DRYAD_DATA_URL:-}" ] && ! has_webdav_source; then
    echo "AADM2025Dryad was not found locally, and no complete download source is configured." >&2
    echo "Provide AADM2025DRYAD_DATA_URL/AADM2025DRYAD_URL, or AADM2025DRYAD_DATA_WEBDAV_URL, AADM2025DRYAD_DATA_KEY, and AADM2025DRYAD_DATA_PASSWORD." >&2
    exit 1
  fi

  staging="${cache_dataset_root}.tmp.${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}.${TEST_FLIGHT:-job}"
  rm -rf "${staging}"
  mkdir -p "${staging}"

  echo "Dataset not found locally; downloading into ${cache_dataset_root}."
  if ! download_direct_archive "${staging}"; then
    if ! has_webdav_source; then
      echo "Direct archive download failed, and no complete WebDAV source is configured." >&2
      rm -rf "${staging}"
      exit 1
    fi
    ensure_rclone
    rm -rf "${staging}"
    mkdir -p "${staging}"
    if ! copy_webdav_to_staging "${staging}"; then
      echo "rclone WebDAV copy failed; trying authenticated archive download fallback." >&2
      rm -rf "${staging}"
      mkdir -p "${staging}"
      if ! download_webdav_archive "${staging}"; then
        echo "Dataset download failed. If this is a password-protected share-page URL, configure AADM2025DRYAD_DATA_WEBDAV_URL as the share WebDAV endpoint or provide a direct ZIP URL via AADM2025DRYAD_DATA_URL." >&2
        rm -rf "${staging}"
        exit 1
      fi
    fi
  fi

  if ! find_rf_root_allow_transient "${staging}" >/dev/null; then
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
