"""Backup system configuration."""

BACKUP_CONFIG = {
    'postgres': {
        'backup_interval_minutes': 15,
        'wal_archive_command': 'aws s3 cp %p s3://genesis-wal/%f',
        'backup_format': 'custom',  # For parallel restore
        'compression_level': 9
    },
    's3': {
        'primary_bucket': 'genesis-backups-primary',
        'regions': ['us-east-1', 'us-west-2'],
        'encryption': 'AES256',
        'storage_class': 'STANDARD_IA',
        'lifecycle_days': 7
    },
    'monitoring': {
        'metrics_port': 9090,
        'alert_webhook': 'https://events.pagerduty.com/v2/enqueue',
        'health_check_interval': 60
    },
    'retention': {
        'hourly_days': 7,
        'daily_days': 30,
        'monthly_days': 365,
        'yearly_forever': True
    },
    'recovery': {
        'target_rto_minutes': 15,
        'target_rpo_minutes': 15,
        'test_recovery_schedule': 'weekly'
    }
}