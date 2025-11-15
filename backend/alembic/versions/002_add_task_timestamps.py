"""Add started_at and finished_at to tasks

Revision ID: 002_add_task_timestamps
Revises: 001
Create Date: 2025-11-15 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "002_add_task_timestamps"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("started_at", sa.DateTime(), nullable=True))
    op.add_column("tasks", sa.Column("finished_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("tasks", "finished_at")
    op.drop_column("tasks", "started_at")

