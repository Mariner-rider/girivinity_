#!/usr/bin/env bash
# Run once on your VPS to create the PostgreSQL user and database.
set -e

DB_NAME="girivinity"
DB_USER="girivinity_user"
DB_PASS="changeme"

echo "Setting up PostgreSQL for Girivinity..."

sudo -u postgres psql <<EOF
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER'
    ) THEN
        CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';
    END IF;
END
$$;
EOF

sudo -u postgres psql <<EOF
SELECT 'CREATE DATABASE $DB_NAME OWNER $DB_USER'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = '$DB_NAME'
)\gexec
EOF

sudo -u postgres psql -d "$DB_NAME" <<EOF
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
GRANT ALL ON SCHEMA public TO $DB_USER;
EOF

echo "PostgreSQL setup complete."
echo "Database: $DB_NAME"
echo "User:     $DB_USER"
echo "Update config.yaml with your actual password before starting."
