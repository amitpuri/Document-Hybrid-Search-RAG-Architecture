#!/bin/bash

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    return 1
fi

while IFS='=' read -r key value || [ -n "$key" ]; do
    # Remove Windows CR
    key="${key//$'\r'/}"
    value="${value//$'\r'/}"

    # Skip blank lines and comments
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue

    # Remove whitespace around key
    key="$(echo "$key" | xargs)"

    # Remove surrounding quotes
    value="${value#\"}"
    value="${value%\"}"
    value="${value#\'}"
    value="${value%\'}"

    export "$key=$value"
done < "$ENV_FILE"

echo "Loaded environment variables from $ENV_FILE"
