#!/usr/bin/env sh
# CORe ML5 installer for Linux.
# - Downloads ml5/ml5d from the latest GitHub release
# - Installs to $PREFIX/bin (default /usr/local/bin)
# - Registers ml5d as a service (systemd, OpenRC, runit, or sysvinit)
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/OpenCORe-Software/ML5/master/install-linux.sh | sudo sh
#   sudo sh install.sh [--prefix DIR] [--no-service] [--version TAG] [--user NAME]
set -eu

REPO="OpenCORe-Software/ML5"
PREFIX="/usr/local"
VERSION="latest"
SERVICE=1
SERVICE_USER="${SUDO_USER:-root}"

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

[ "$(id -u)" -eq 0 ] || die "Run as root (e.g. sudo sh install.sh)"
uname -s | grep -qi linux || die "This script is for Linux. Use install-macos.sh on macOS."

case "$(uname -m)" in
    x86_64|amd64) ;;
    *) die "Unsupported architecture: $(uname -m) (release assets are x86_64 only)" ;;
esac

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q -O "$2" "$1"; }
else
    die "Need curl or wget to download release assets."
fi

if [ "$VERSION" = "latest" ]; then
    BASE="https://github.com/$REPO/releases/latest/download"
else
    BASE="https://github.com/$REPO/releases/download/$VERSION"
fi

BINDIR="$PREFIX/bin"
mkdir -p "$BINDIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

info "Downloading ml5/ml5d from $BASE ..."
fetch "$BASE/ml5-linux"  "$TMP/ml5"
fetch "$BASE/ml5d-linux" "$TMP/ml5d"
chmod 755 "$TMP/ml5" "$TMP/ml5d"

install -m 755 "$TMP/ml5"  "$BINDIR/ml5"
install -m 755 "$TMP/ml5d" "$BINDIR/ml5d"
ok "Installed binaries to $BINDIR"

[ "$SERVICE" -eq 1 ] || { ok "Skipping service setup (--no-service)."; exit 0; }

HOME_DIR="$(getent passwd "$SERVICE_USER" 2>/dev/null | cut -d: -f6 || true)"
[ -n "$HOME_DIR" ] || HOME_DIR="$(eval echo "~$SERVICE_USER" 2>/dev/null || echo /root)"
LOG_DIR="$HOME_DIR/.ml5/logs"
mkdir -p "$LOG_DIR"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$HOME_DIR/.ml5" 2>/dev/null || true

install_systemd() {
    cat > /etc/systemd/system/ml5d.service <<EOF
[Unit]
Description=CORe ML5 inference daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
ExecStart=$BINDIR/ml5d
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now ml5d.service
    ok "systemd service 'ml5d' enabled and started."
}

install_openrc() {
    cat > /etc/init.d/ml5d <<EOF
#!/sbin/openrc-run
name="ml5d"
description="CORe ML5 inference daemon"
command="$BINDIR/ml5d"
command_user="$SERVICE_USER"
command_background="yes"
pidfile="/run/ml5d.pid"
output_log="$LOG_DIR/ml5d.log"
error_log="$LOG_DIR/ml5d.log"
depend() {
    need net
}
EOF
    chmod 755 /etc/init.d/ml5d
    rc-update add ml5d default
    rc-service ml5d restart
    ok "OpenRC service 'ml5d' enabled and started."
}

install_runit() {
    SV_DIR="$( [ -d /etc/runit/sv ] && echo /etc/runit/sv || echo /etc/sv )"
    mkdir -p "$SV_DIR/ml5d/log"
    cat > "$SV_DIR/ml5d/run" <<EOF
#!/bin/sh
exec chpst -u $SERVICE_USER $BINDIR/ml5d 2>&1
EOF
    cat > "$SV_DIR/ml5d/log/run" <<EOF
#!/bin/sh
exec svlogd -tt $LOG_DIR
EOF
    chmod 755 "$SV_DIR/ml5d/run" "$SV_DIR/ml5d/log/run"
    if [ -d /var/service ]; then
        ln -sfn "$SV_DIR/ml5d" /var/service/ml5d
    elif [ -d /etc/runit/runsvdir/default ]; then
        ln -sfn "$SV_DIR/ml5d" /etc/runit/runsvdir/default/ml5d
    fi
    ok "runit service 'ml5d' installed (linked into runsvdir)."
}

install_sysvinit() {
    cat > /etc/init.d/ml5d <<'EOF'
#!/bin/sh
### BEGIN INIT INFO
# Provides:          ml5d
# Required-Start:    $network
# Required-Stop:     $network
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: CORe ML5 inference daemon
### END INIT INFO
EOF
    cat >> /etc/init.d/ml5d <<EOF
DAEMON=$BINDIR/ml5d
USER=$SERVICE_USER
PIDFILE=/var/run/ml5d.pid
LOG=$LOG_DIR/ml5d.log

case "\$1" in
    start)
        echo "Starting ml5d"
        start-stop-daemon --start --background --make-pidfile --pidfile "\$PIDFILE" \\
            --chuid "\$USER" --exec "\$DAEMON" >> "\$LOG" 2>&1
        ;;
    stop)
        echo "Stopping ml5d"
        start-stop-daemon --stop --pidfile "\$PIDFILE" --retry 5
        rm -f "\$PIDFILE"
        ;;
    restart)
        "\$0" stop; "\$0" start
        ;;
    status)
        if [ -f "\$PIDFILE" ] && kill -0 "\$(cat "\$PIDFILE")" 2>/dev/null; then
            echo "ml5d running (pid \$(cat "\$PIDFILE"))"
        else
            echo "ml5d not running"; exit 3
        fi
        ;;
    *) echo "Usage: \$0 {start|stop|restart|status}"; exit 1 ;;
esac
EOF
    chmod 755 /etc/init.d/ml5d
    if command -v update-rc.d >/dev/null 2>&1; then
        update-rc.d ml5d defaults
    elif command -v chkconfig >/dev/null 2>&1; then
        chkconfig --add ml5d
    fi
    service ml5d start 2>/dev/null || /etc/init.d/ml5d start
    ok "sysvinit service 'ml5d' enabled and started."
}

if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    install_systemd
elif command -v openrc >/dev/null 2>&1 || { [ -d /etc/init.d ] && [ -f /sbin/openrc-run ]; }; then
    install_openrc
elif command -v sv >/dev/null 2>&1 && { [ -d /etc/runit/sv ] || [ -d /etc/sv ]; }; then
    install_runit
elif [ -d /etc/init.d ]; then
    install_sysvinit
else
    warn "No supported init system found (systemd/OpenRC/runit/sysvinit)."
    warn "Binaries are installed; run 'ml5d --background' to start the daemon manually."
    exit 0
fi

ok "Done. Try:  ml5 status"
