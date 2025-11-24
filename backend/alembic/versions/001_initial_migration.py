"""Initial migration - create uploads/tasks and related tables

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

    # 创建 uploads 表（输入文件表，用于去重存储上传视频的元信息）
    op.create_table(
        'uploads',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('size', sa.BigInteger(), nullable=True),
        sa.Column('sha256', sa.String(length=64), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('fps', sa.Integer(), nullable=True),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('total_frames', sa.Integer(), nullable=True),
        sa.Column('bit_rate', sa.BigInteger(), nullable=True),
        sa.Column('avg_frame_rate', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_uploads_id'), 'uploads', ['id'], unique=False)
    op.create_index(op.f('ix_uploads_file_path'), 'uploads', ['file_path'], unique=True)
    op.create_index(op.f('ix_uploads_sha256'), 'uploads', ['sha256'], unique=True)

    # 创建 tasks 表（聚焦调度与状态，通过 upload_id 关联 uploads）
    op.create_table(
        'tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('status', task_status_enum, nullable=False),
        sa.Column('upload_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('output_file_path', sa.String(length=500), nullable=True),
        sa.Column('output_file_name', sa.String(length=255), nullable=True),
        sa.Column('progress', sa.Float(), nullable=False),
        sa.Column('total_frames', sa.Integer(), nullable=True),
        sa.Column('processed_frames', sa.Integer(), nullable=False),
        sa.Column('estimated_time_remaining', sa.Integer(), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['upload_id'], ['uploads.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
    )
    
    # 创建索引
    op.create_index(op.f('ix_tasks_id'), 'tasks', ['id'], unique=False)
    op.create_index(op.f('ix_tasks_status'), 'tasks', ['status'], unique=False)
    op.create_index(op.f('ix_tasks_celery_task_id'), 'tasks', ['celery_task_id'], unique=False)
    op.create_index(op.f('ix_tasks_upload_id'), 'tasks', ['upload_id'], unique=False)

    # 创建 task_parameters 表（结构化参数 + 高级参数 JSON）
    op.create_table(
        'task_parameters',
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('upload_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('scale', sa.Float(), nullable=False),
        sa.Column('sparse_ratio', sa.Float(), nullable=False),
        sa.Column('local_range', sa.Integer(), nullable=False),
        sa.Column('seed', sa.Integer(), nullable=False),
        sa.Column('model_variant', sa.String(length=50), nullable=False),
        sa.Column('preprocess_width', sa.Integer(), nullable=False),
        sa.Column('preserve_aspect_ratio', sa.Boolean(), nullable=False),
        sa.Column('advanced_options', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['upload_id'], ['uploads.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('task_id'),
        sa.UniqueConstraint(
            'upload_id',
            'scale',
            'sparse_ratio',
            'local_range',
            'seed',
            'model_variant',
            'preprocess_width',
            'preserve_aspect_ratio',
            name='uq_task_parameters_upload_config',
        ),
    )
    op.create_index(
        op.f('ix_task_parameters_upload_id'),
        'task_parameters',
        ['upload_id'],
        unique=False,
    )

    # 创建 task_video_info 表（任务级视频信息）
    op.create_table(
        'task_video_info',
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('total_frames', sa.Integer(), nullable=True),
        sa.Column('fps', sa.Integer(), nullable=True),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('processed_frames', sa.Integer(), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('inference_time', sa.Float(), nullable=True),
        sa.Column('bit_rate', sa.BigInteger(), nullable=True),
        sa.Column('avg_frame_rate', sa.Float(), nullable=True),
        sa.Column('preprocess_applied', sa.Boolean(), nullable=True),
        sa.Column('preprocess_width', sa.Integer(), nullable=True),
        sa.Column('preprocess_result_width', sa.Integer(), nullable=True),
        sa.Column('preprocess_result_height', sa.Integer(), nullable=True),
        sa.Column('predicted_output_width', sa.Integer(), nullable=True),
        sa.Column('predicted_output_height', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('task_id'),
    )


def downgrade() -> None:
    op.drop_table('task_video_info')
    op.drop_index(op.f('ix_task_parameters_upload_id'), table_name='task_parameters')
    op.drop_table('task_parameters')
    op.drop_index(op.f('ix_tasks_celery_task_id'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_status'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_id'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_upload_id'), table_name='tasks')
    op.drop_table('tasks')
    op.drop_index(op.f('ix_uploads_sha256'), table_name='uploads')
    op.drop_index(op.f('ix_uploads_file_path'), table_name='uploads')
    op.drop_index(op.f('ix_uploads_id'), table_name='uploads')
    op.drop_table('uploads')
    task_status_enum.drop(op.get_bind(), checkfirst=True)
