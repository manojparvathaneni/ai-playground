#!/usr/bin/env bash
# Restore Phoenix and Langfuse data from a backup made by docker/backup.sh,
# on this machine or another one.
#
# Usage:  make restore BACKUP=backups/<date-time>
#    (or: docker/restore.sh backups/<date-time>)
#
# This REPLACES the current Phoenix and Langfuse data with the backup's, so
# it asks before changing anything.

set -euo pipefail
cd "$(dirname "$0")/.."

HELPER_IMAGE=alpine:3.22

backup_dir=${1:-}
if [ -z "$backup_dir" ] || [ ! -d "$backup_dir" ]; then
    echo "Usage: make restore BACKUP=backups/<date-time>"
    echo "Backups found:"
    ls -1d backups/*/ 2>/dev/null || echo "  (none)"
    exit 1
fi

project=$(docker compose config | sed -n 's/^name: //p')
volumes=$(docker compose config --volumes)

# --- Check everything before changing anything ---------------------------

for name in $volumes; do
    if [ ! -f "$backup_dir/$name.tar.gz" ]; then
        echo "The backup has no $name.tar.gz, so it can't be restored."
        exit 1
    fi
done

# The data only works with the secrets it was created with.
if [ -f "$backup_dir/.env" ]; then
    if [ ! -f .env ]; then
        cp "$backup_dir/.env" .env
        chmod 600 .env
        echo "Copied .env from the backup."
    elif ! cmp -s .env "$backup_dir/.env"; then
        echo "Your .env is different from the one in the backup."
        echo "Langfuse's data only works with the secrets it was created with."
        echo "To use the backup's, run:  cp $backup_dir/.env .env"
        echo "then run the restore again."
        exit 1
    fi
fi

# A database's files only work with the version that wrote them.
if [ -f "$backup_dir/images.txt" ] &&
    ! diff -q <(sort "$backup_dir/images.txt") <(docker compose config --images | sort) > /dev/null; then
    echo "Warning: the image versions differ from the ones the backup was made with:"
    diff <(sort "$backup_dir/images.txt") <(docker compose config --images | sort) || true
    echo "If a database doesn't start, use the backup's versions, then upgrade."
fi

echo "This replaces the current Phoenix and Langfuse data with $backup_dir."
read -r -p "Type yes to continue: " answer
if [ "$answer" != "yes" ]; then
    echo "Nothing changed."
    exit 1
fi

# --- Restore ---------------------------------------------------------------

# Stop the containers, and create any volumes that don't exist yet (on a new
# machine, none do), without starting anything.
docker compose --progress quiet stop
docker compose --progress quiet create

for name in $volumes; do
    volume="${project}_${name}"
    # The helper container empties the volume, then unpacks the archive,
    # which it reads from its input, into it.
    docker run --rm --interactive --volume "$volume:/data" "$HELPER_IMAGE" \
        sh -c 'find /data -mindepth 1 -delete && tar -xzf - --numeric-owner -C /data' \
        < "$backup_dir/$name.tar.gz"
    echo "  restored $volume"
done

docker compose --progress quiet up -d
echo
echo "Restored. Phoenix: http://localhost:6006  Langfuse: http://localhost:3000"
