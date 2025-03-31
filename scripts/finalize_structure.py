#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_move(src, dst):
    """Safely move a file or directory."""
    try:
        if os.path.exists(src):
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            
            # If destination exists, remove it first
            if os.path.exists(dst):
                if os.path.isfile(dst):
                    os.remove(dst)
                else:
                    shutil.rmtree(dst)
            
            # Move the file or directory
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                os.remove(src)
            else:
                shutil.copytree(src, dst)
                shutil.rmtree(src)
            print(f"Moved {src} to {dst}")
    except Exception as e:
        print(f"Error moving {src} to {dst}: {e}")

def move_python_files():
    """Move Python files to their correct locations."""
    moves = [
        ('run.py', 'backend/main.py'),
        ('run_investment_system.py', 'trading_engine/main.py'),
        ('train_sentiment_models.py', 'trading_engine/models/sentiment.py'),
        ('launch.py', 'scripts/deployment/launch.py'),
        ('setup_modular_dashboard.py', 'scripts/setup/setup_dashboard.py')
    ]
    
    for src, dst in moves:
        safe_move(src, dst)

def organize_config_files():
    """Organize configuration files."""
    # Move to backend
    backend_configs = ['pytest.ini', 'setup.py', 'anaconda-project.yml']
    for config in backend_configs:
        safe_move(config, f'backend/{config}')
    
    # Keep in root
    root_configs = ['.env', '.env.template', 'requirements.txt', 'package.json', 'docker-compose.yml']
    # These files are already in root, no need to move them

def organize_documentation():
    """Organize documentation files."""
    moves = [
        ('CHANGELOG.md', 'docs/CHANGELOG.md'),
        ('REPOSITORY-STRUCTURE.md', 'docs/architecture/REPOSITORY-STRUCTURE.md'),
        ('PLACEHOLDER-MANAGEMENT.md', 'docs/guides/PLACEHOLDER-MANAGEMENT.md'),
        ('PLACEHOLDERS.md', 'docs/guides/PLACEHOLDERS.md')
    ]
    
    for src, dst in moves:
        safe_move(src, dst)

def clean_up_logs():
    """Move log files to data_pipeline/logs/."""
    log_dir = 'data_pipeline/logs'
    ensure_dir(log_dir)
    
    for log_file in Path().glob('*.log'):
        safe_move(str(log_file), f'{log_dir}/{log_file.name}')

def move_test_files():
    """Move test files to backend/tests/."""
    if os.path.exists('tests'):
        safe_move('tests', 'backend/tests')

def clean_up_unnecessary():
    """Remove unnecessary files and directories."""
    to_remove = [
        'client',  # Already moved to frontend
        'packages',  # Not needed with new structure
        '.DS_Store',
        'package-lock.json',  # Using pnpm
        'pnpm-workspace.yaml',  # Not needed
        'turbo.json'  # Not needed
    ]
    
    for item in to_remove:
        try:
            path = Path(item)
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                print(f"Removed {item}")
        except Exception as e:
            print(f"Error removing {item}: {e}")

def main():
    """Main function to finalize repository structure."""
    print("Starting final repository structure organization...")
    
    # Create all necessary directories
    for dir_path in [
        'backend/api',
        'backend/models',
        'backend/services',
        'backend/config',
        'frontend/components',
        'frontend/pages',
        'frontend/public',
        'trading_engine/models',
        'trading_engine/strategies',
        'trading_engine/backtesting',
        'data_pipeline/connectors',
        'data_pipeline/processors',
        'data_pipeline/storage',
        'data_pipeline/logs',
        'docs/api',
        'docs/architecture',
        'docs/guides',
        'scripts/deployment',
        'scripts/setup',
        'scripts/maintenance'
    ]:
        ensure_dir(dir_path)
    
    # Move files to their correct locations
    print("\nMoving Python files...")
    move_python_files()
    
    print("\nOrganizing configuration files...")
    organize_config_files()
    
    print("\nOrganizing documentation...")
    organize_documentation()
    
    print("\nCleaning up logs...")
    clean_up_logs()
    
    print("\nMoving test files...")
    move_test_files()
    
    print("\nCleaning up unnecessary files...")
    clean_up_unnecessary()
    
    print("\nRepository structure organization complete!")

if __name__ == "__main__":
    main() 