#!/bin/sh
# CORe ML5 installer for macOS.
# - Downloads ml5/ml5d from the latest GitHub release
# - Installs to $PREFIX/bin (default /usr/local/bin)
# - Registers ml5d as a launchd LaunchDaemon (runs at boot, keeps alive)
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/OpenCORe-Software/ML5/main/install-macos.sh | sudo sh
#   sudo sh install-macos.sh [--prefix DIR] [--no-service] [--version TAG] [--user NAME]
set -eu

REPO="OpenCORe-Software/ML5"
PREFIX="/usr/local"
VERSION="latest"
SERVICE=1
SERVICE_USER="${SUDO_USER:-root}"
LABEL="com.opencore.ml5d"
PLIST="/Library/LaunchDaemons/$LABEL.plist"

info() { printf '\033[36m[ml5]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ml5]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[ml5]\033[0m %s\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)     PREFIX="$2"; shift 2 ;;
        --version)    VERSION="$2"; shift 2 ;;
        --user)       SERVICE_USER="$2"; shift 2 ;;
        --no-service) SERVICE=0; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[ "$(id -u)" -eq 0 ] || die "Run as root (e.g. sudo sh install-macos.sh)"
[ "$(uname -s)" = "Darwin" ] || die "This script is for macOS. Use install.sh on Linux."

case "$(uname -m)" in
    x86_64|arm64) ;;
    *) die "Unsupported architecture: $(uname -m)" ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required (ships with macOS)."
fetch() { curl -fSL --retry 3 -o "$2" "$1"; }

if [ "$VERSION" = "latest" ]; then
    BASE="https://github.com/$REPO/releases/latest/download"
else
    BASE="https://github.com/$REPO/releases/download/$VERSION"
fi

# /usr/local is not writable even for root on some systems without prep;
# on Apple Silicon Homebrew owns it. Fall back is left to the user via --prefix.
BINDIR="$PREFIX/bin"
mkdir -p "$BINDIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

info "Downloading ml5/ml5d from $BASE ..."
fetch "$BASE/ml5-macos"  "$TMP/ml5"
fetch "$BASE/ml5d-macos" "$TMP/ml5d"
chmod 755 "$TMP/ml5" "$TMP/ml5d"

# Remove quarantine attribute if present (files from curl usually don't get it,
# but be defensive in case the archive was re-hosted).
xattr -d com.apple.quarantine "$TMP/ml5"  2>/dev/null || true
xattr -d com.apple.quarantine "$TMP/ml5d" 2>/dev/null || true

install -m 755 "$TMP/ml5"  "$BINDIR/ml5"
install -m 755 "$TMP/ml5d" "$BINDIR/ml5d"
ok "Installed binaries to $BINDIR"

[ "$SERVICE" -eq 1 ] || { ok "Skipping service setup (--no-service)."; exit 0; }

HOME_DIR="$(dscl . -read "/Users/$SERVICE_USER" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
[ -n "$HOME_DIR" ] || HOME_DIR="/var/root"
LOG_DIR="$HOME_DIR/.ml5/logs"
mkdir -p "$LOG_DIR"
chown -R "$SERVICE_USER" "$HOME_DIR/.ml5" 2>/dev/null || true

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$BINDIR/ml5d</string>
    </array>
    <key>UserName</key>
    <string>$SERVICE_USER</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/ml5d.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/ml5d.log</string>
</dict>
</plist>
EOF
chmod 644 "$PLIST"
chown root:wheel "$PLIST"

# Unload any previous install, then load fresh.
launchctl bootout system "$PLIST" 2>/dev/null || true
launchctl bootstrap system "$PLIST"
launchctl enable "system/$LABEL"
launchctl kickstart -k "system/$LABEL"
ok "launchd daemon '$LABEL' loaded and started."

ok "Done. Try:  ml5 status"
