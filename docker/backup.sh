#!/usr/bin/env bash
# Back up everything Phoenix and Langfuse have stored, so it can be restored
# later or on another machine with docker/restore.sh.
#
# Usage:  make backup   (or: docker/backup.sh)
#
# Creates backups/<date-time>/ with:
#   <volume>.tar.gz   one archive per Docker volume (traces, scores, users...)
#   .env              the secrets the data was created with; restore needs them
#   images.txt        the image versions the data was created with
#
# The containers are stopped while the archives are made, so no database is
# half-way through writing a file, and started again afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

# Small image used only to read the volumes. Pinned, like the other images.
HELPER_IMAGE=alpine:3.22

# Docker names each volume "<project>_<name in the compose file>".
project=$(docker compose config | sed -n 's/^name: //p')
volumes=$(docker compose config --volumes)

backup_dir="backups/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$backup_dir"

# Stop the containers if they're running, and start them again at the end,
# even if a step below fails.
if [ -n "$(docker compose ps --quiet)" ]; then
    echo "Stopping the containers..."
    docker compose --progress quiet stop
    trap 'echo "Starting the containers again..."; docker compose --progress quiet start' EXIT
fi

for name in $volumes; do
    volume="${project}_${name}"
    if ! docker volume inspect "$volume" > /dev/null 2>&1; then
        echo "  skip $volume (it doesn't exist yet)"
        continue
    fi
    # The helper container sees the volume at /data, read-only, and writes
    # the archive to its output, which we save to a file. --numeric-owner
    # keeps each file's owner, which databases such as Postgres check.
    docker run --rm --volume "$volume:/data:ro" "$HELPER_IMAGE" \
        tar -czf - --numeric-owner -C /data . > "$backup_dir/$name.tar.gz"
    echo "  saved $volume"
done

# Langfuse encrypts some data with keys from .env, and the databases expect
# its passwords, so the data can't be used without it.
if [ -f .env ]; then
    cp .env "$backup_dir/.env"
fi
docker compose config --images > "$backup_dir/images.txt"
chmod -R go-rwx "$backup_dir"  # it contains secrets: readable only by you

echo
echo "Backup saved to $backup_dir ($(du -sh "$backup_dir" | cut -f1))."
echo "It includes .env, so keep it private, like a password."
