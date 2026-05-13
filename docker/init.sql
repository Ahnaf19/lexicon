-- Pre-create extensions as superuser so Alembic migrations can run as the app user.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
