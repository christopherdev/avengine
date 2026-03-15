from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Alembic config ────────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Import ORM metadata ───────────────────────────────────────────────────────
from src.core.database import Base  # noqa: E402
from src.core.models import Task    # noqa: E402, F401  (import to register with Base)

target_metadata = Base.metadata

# ── Database URL (prefer env var over alembic.ini) ────────────────────────────
def _get_url() -> str:
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url", ""))
    # Alembic uses the sync psycopg2 driver; strip the +asyncpg suffix if present
    return url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")


# ── Offline mode ──────────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode ───────────────────────────────────────────────────────────────
def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
