#!/usr/bin/env python3
"""
Script to manage personal content visibility in the repository.
This allows you to keep personal articles locally while excluding them from public repos.
"""

import os
import shutil
import argparse
from pathlib import Path

PERSONAL_FILES = [
    "docs/my_whale_ltv_journey.md",
    "docs/executive_summary.md"
]

BACKUP_DIR = "personal_content_backup"

def backup_personal_files():
    """Backup personal files to a separate directory."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    for file_path in PERSONAL_FILES:
        if os.path.exists(file_path):
            backup_path = os.path.join(BACKUP_DIR, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            print(f"✅ Backed up: {file_path} -> {backup_path}")
        else:
            print(f"⚠️  File not found: {file_path}")

def restore_personal_files():
    """Restore personal files from backup."""
    for file_path in PERSONAL_FILES:
        backup_path = os.path.join(BACKUP_DIR, os.path.basename(file_path))
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print(f"✅ Restored: {backup_path} -> {file_path}")
        else:
            print(f"⚠️  Backup not found: {backup_path}")

def check_git_status():
    """Check if personal files are being tracked by git."""
    import subprocess
    
    for file_path in PERSONAL_FILES:
        if os.path.exists(file_path):
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain", file_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.stdout.strip():
                    print(f"⚠️  {file_path} is tracked by git")
                else:
                    print(f"✅ {file_path} is ignored by git")
            except subprocess.CalledProcessError:
                print(f"❓ Could not check git status for {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Manage personal content visibility")
    parser.add_argument(
        "action",
        choices=["backup", "restore", "check"],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    if args.action == "backup":
        print("📦 Backing up personal files...")
        backup_personal_files()
        print("\n💡 These files are now backed up and excluded from git.")
        print("   They won't be pushed to the public repository.")
        
    elif args.action == "restore":
        print("📥 Restoring personal files...")
        restore_personal_files()
        print("\n💡 Personal files restored. Remember they're still in .gitignore.")
        
    elif args.action == "check":
        print("🔍 Checking personal files status...")
        check_git_status()
        print("\n📋 Personal files status:")
        for file_path in PERSONAL_FILES:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path} exists locally")
            else:
                print(f"   ❌ {file_path} not found")

if __name__ == "__main__":
    main() 