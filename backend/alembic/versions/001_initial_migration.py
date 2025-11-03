"""Initial migration - create tasks table

Revision ID: 001
Revises: 
Create Date: 2025-11-03 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

task_status_enum = postgresql.ENUM('pending', 'processing', 'completed', 'failed', name='taskstatus')


def upgrade() -> None:
    # 创建任务状态枚举类型
    task_status_enum.create(op.get_bind(), checkfirst=True)

    # 创建tasks表
    op.create_table(
        'tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('status', task_status_enum, nullable=False),
        sa.Column('input_file_path', sa.String(length=500), nullable=False),
        sa.Column('input_file_name', sa.String(length=255), nullable=False),
        sa.Column('output_file_path', sa.String(length=500), nullable=True),
        sa.Column('output_file_name', sa.String(length=255), nullable=True),
        sa.Column('video_info', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('parameters', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('progress', sa.Float(), nullable=False),
        sa.Column('total_frames', sa.Integer(), nullable=True),
        sa.Column('processed_frames', sa.Integer(), nullable=False),
        sa.Column('estimated_time_remaining', sa.Integer(), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建索引
    op.create_index(op.f('ix_tasks_id'), 'tasks', ['id'], unique=False)
    op.create_index(op.f('ix_tasks_status'), 'tasks', ['status'], unique=False)
    op.create_index(op.f('ix_tasks_celery_task_id'), 'tasks', ['celery_task_id'], unique=False)
def downgrade() -> None:
    op.drop_index(op.f('ix_tasks_celery_task_id'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_status'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_id'), table_name='tasks')
    op.drop_table('tasks')
    task_status_enum.drop(op.get_bind(), checkfirst=True)
