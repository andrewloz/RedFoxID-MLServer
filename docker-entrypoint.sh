#!/bin/sh
set -e

# Default to the repository config if no arguments are supplied.
if [ "$#" -eq 0 ]; then
  set -- config.ini
fi

case "$1" in
  python|bash|sh)
    exec "$@"
    ;;
  -*)
    exec python server.py "$@"
    ;;
  *)
    exec python server.py "$@"
    ;;
esac
