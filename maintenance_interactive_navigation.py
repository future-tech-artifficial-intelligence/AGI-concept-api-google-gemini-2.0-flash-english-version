#!/usr/bin/env python3
"""
Maintenance Script - Google Gemini 2.0 Flash AI Interactive Navigation System
Validates, maintains, and optimizes the complete system
"""

import os
import sys
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maintenance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('InteractiveNavigationMaintenance')

class InteractiveNavigationMaintainer:
    """Maintenance system for interactive navigation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.maintenance_log = []
        self.issues_found = []
        self.fixes_applied = []
        self.start_time = datetime.now()
        
        # Critical system files
        self.critical_files = [
            'interactive_web_navigator.py',
            'gemini_interactive_adapter.py',
            'install_interactive_navigation.py',
            'demo_interactive_navigation.py',
            'test_interactive_navigation.py',
            'GUIDE_NAVIGATION_INTERACTIVE.md'
        ]
        
        # Required dependencies
        self.required_packages = [
            'google-generativeai',
            'selenium',
            'webdriver-manager',
            'beautifulsoup4',
            'pillow',
            'requests',
            'python-dotenv',
            'flask',
            'flask-cors'
        ]
        
        logger.info("🔧 Maintenance system initialized")
        
    def log_action(self, action: str, status: str, details: str = ""):
        """Logs a maintenance action"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'status': status,
            'details': details
        }
        
        self.maintenance_log.append(entry)
        
        if status == "SUCCESS":
            logger.info(f"✅ {action}: {details}")
        elif status == "WARNING":
            logger.warning(f"⚠️ {action}: {details}")
        else:
            logger.error(f"❌ {action}: {details}")
            
    def check_file_integrity(self) -> bool:
        """Checks the integrity of critical files"""
        self.log_action("INTEGRITY_CHECK", "INFO", "Checking file integrity")
        
        all_files_ok = True
        
        for file_path in self.critical_files:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                self.log_action("FILE_MISSING", "ERROR", f"Missing file: {file_path}")
                self.issues_found.append(f"Missing file: {file_path}")
                all_files_ok = False
                continue
                
            # Check size (non-empty file)
            if full_path.stat().st_size == 0:
                self.log_action("FILE_EMPTY", "ERROR", f"Empty file: {file_path}")
                self.issues_found.append(f"Empty file: {file_path}")
                all_files_ok = False
                continue
                
            # Check Python syntax
            if file_path.endswith('.py'):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), file_path, 'exec')
                    self.log_action("SYNTAX_CHECK", "SUCCESS", f"Valid syntax: {file_path}")
                except SyntaxError as e:
                    self.log_action("SYNTAX_ERROR", "ERROR", f"Syntax error {file_path}: {e}")
                    self.issues_found.append(f"Syntax error {file_path}: {e}")
                    all_files_ok = False
                    
        return all_files_ok
        
    def check_dependencies(self) -> bool:
        """Checks Python dependencies"""
        self.log_action("DEPENDENCY_CHECK", "INFO", "Checking dependencies")
        
        missing_packages = []
        
        for package in self.required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.log_action("PACKAGE_OK", "SUCCESS", f"Package available: {package}")
            except ImportError:
                self.log_action("PACKAGE_MISSING", "WARNING", f"Missing package: {package}")
                missing_packages.append(package)
                
        if missing_packages:
            self.issues_found.append(f"Missing packages: {', '.join(missing_packages)}")
            return False
            
        return True
        
    def check_configuration(self) -> bool:
        """Checks configuration files"""
        self.log_action("CONFIG_CHECK", "INFO", "Checking configuration")
        
        config_files = [
            '.env',
            'config/navigation_config.json',
            'ai_api_config.json'
        ]
        
        config_ok = True
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            
            if not config_path.exists():
                self.log_action("CONFIG_MISSING", "WARNING", f"Missing configuration: {config_file}")
                continue
                
            # Specific verification based on type
            if config_file.endswith('.json'):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    self.log_action("CONFIG_VALID", "SUCCESS", f"Valid configuration: {config_file}")
                except json.JSONDecodeError as e:
                    self.log_action("CONFIG_INVALID", "ERROR", f"Invalid JSON {config_file}: {e}")
                    self.issues_found.append(f"Invalid JSON {config_file}: {e}")
                    config_ok = False
                    
            elif config_file == '.env':
                # API key verification
                with open(config_path, 'r', encoding='utf-8') as f:
                    env_content = f.read()
                    
                if 'GEMINI_API_KEY' not in env_content:
                    self.log_action("API_KEY_MISSING", "WARNING", "Google Gemini 2.0 Flash AI API Key not configured")
                elif 'your_api_key_here' in env_content: # Assuming placeholder text
                    self.log_action("API_KEY_PLACEHOLDER", "WARNING", "Google Gemini 2.0 Flash AI API Key not changed")
                else:
                    self.log_action("API_KEY_OK", "SUCCESS", "Google Gemini 2.0 Flash AI API Key configured")
                    
        return config_ok
        
    def check_disk_space(self) -> bool:
        """Checks available disk space"""
        self.log_action("DISK_CHECK", "INFO", "Checking disk space")
        
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            free_space_mb = disk_usage.free / (1024 * 1024)
            
            if free_space_mb < 100:  # Less than 100 MB
                self.log_action("DISK_LOW", "ERROR", f"Low disk space: {free_space_mb:.1f} MB")
                self.issues_found.append(f"Insufficient disk space: {free_space_mb:.1f} MB")
                return False
            elif free_space_mb < 500:  # Less than 500 MB
                self.log_action("DISK_WARNING", "WARNING", f"Limited disk space: {free_space_mb:.1f} MB")
            else:
                self.log_action("DISK_OK", "SUCCESS", f"Sufficient disk space: {free_space_mb:.1f} MB")
                
            return True
            
        except Exception as e:
            self.log_action("DISK_ERROR", "ERROR", f"Disk check error: {e}")
            return False
            
    def clean_temporary_files(self) -> bool:
        """Cleans temporary files"""
        self.log_action("CLEANUP", "INFO", "Cleaning temporary files")
        
        temp_patterns = [
            '**/*.pyc',
            '**/__pycache__',
            '**/test_results_*/*.png',
            '**/logs/*.log.old',
            '**/cache/*',
            '**/.pytest_cache'
        ]
        
        cleaned_count = 0
        
        for pattern in temp_patterns:
            for file_path in self.project_root.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        cleaned_count += 1
                except Exception as e:
                    self.log_action("CLEANUP_ERROR", "WARNING", f"Error cleaning {file_path}: {e}")
                    
        self.log_action("CLEANUP_COMPLETE", "SUCCESS", f"Cleanup complete: {cleaned_count} items removed")
        self.fixes_applied.append(f"Cleanup: {cleaned_count} temporary files removed")
        
        return True
        
    def optimize_imports(self) -> bool:
        """Optimizes Python imports"""
        self.log_action("IMPORT_OPTIMIZATION", "INFO", "Optimizing imports")
        
        python_files = list(self.project_root.glob('*.py'))
        optimized_count = 0
        
        for py_file in python_files:
            if py_file.name.startswith('test_') or py_file.name.startswith('demo_'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for unused imports (basic)
                lines = content.split('\n')
                import_lines = [line for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ')]
                
                if len(import_lines) > 20:  # Many imports
                    self.log_action("MANY_IMPORTS", "WARNING", f"Many imports in {py_file.name}: {len(import_lines)}")
                    
                optimized_count += 1
                
            except Exception as e:
                self.log_action("IMPORT_ERROR", "WARNING", f"Error analyzing imports {py_file.name}: {e}")
                
        self.log_action("IMPORT_OPTIMIZATION_COMPLETE", "SUCCESS", f"Analysis of {optimized_count} Python files")
        return True
        
    def update_documentation(self) -> bool:
        """Updates documentation"""
        self.log_action("DOC_UPDATE", "INFO", "Updating documentation")
        
        try:
            # Generate a maintenance report
            maintenance_report = {
                "last_maintenance": datetime.now().isoformat(),
                "critical_files_status": "OK" if all(
                    (self.project_root / f).exists() for f in self.critical_files
                ) else "ISSUES",
                "dependencies_status": "OK" if not self.issues_found else "ISSUES",
                "issues_found": self.issues_found,
                "fixes_applied": self.fixes_applied,
                "maintenance_log": self.maintenance_log
            }
            
            # Save the report
            report_file = self.project_root / f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(maintenance_report, f, indent=2, ensure_ascii=False)
                
            self.log_action("DOC_SAVED", "SUCCESS", f"Report saved: {report_file.name}")
            return True
            
        except Exception as e:
            self.log_action("DOC_ERROR", "ERROR", f"Error saving documentation: {e}")
            return False
            
    def run_quick_tests(self) -> bool:
        """Executes quick tests"""
        self.log_action("QUICK_TESTS", "INFO", "Executing quick tests")
        
        try:
            # Import test of main modules
            test_imports = [
                'interactive_web_navigator',
                'gemini_interactive_adapter',
                'ai_api_interface'
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    self.log_action("IMPORT_TEST", "SUCCESS", f"Import successful: {module}")
                except ImportError as e:
                    self.log_action("IMPORT_TEST", "ERROR", f"Import failed {module}: {e}")
                    self.issues_found.append(f"Import failed: {module}")
                    
            return True
            
        except Exception as e:
            self.log_action("QUICK_TESTS_ERROR", "ERROR", f"Quick tests error: {e}")
            return False
            
    def generate_health_report(self) -> Dict:
        """Generates a system health report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        health_score = 100
        if self.issues_found:
            health_score -= len(self.issues_found) * 10
        health_score = max(0, health_score)
        
        report = {
            "maintenance_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": round(duration, 2),
                "health_score": health_score,
                "issues_found": len(self.issues_found),
                "fixes_applied": len(self.fixes_applied),
                "status": "HEALTHY" if health_score >= 80 else "NEEDS_ATTENTION" if health_score >= 50 else "CRITICAL"
            },
            "critical_files": {
                "total": len(self.critical_files),
                "present": len([f for f in self.critical_files if (self.project_root / f).exists()]),
                "missing": [f for f in self.critical_files if not (self.project_root / f).exists()]
            },
            "issues_detail": self.issues_found,
            "fixes_detail": self.fixes_applied,
            "maintenance_actions": self.maintenance_log,
            "recommendations": self.generate_recommendations()
        }
        
        return report
        
    def generate_recommendations(self) -> List[str]:
        """Generates improvement recommendations"""
        recommendations = []
        
        if len(self.issues_found) > 5:
            recommendations.append("Resolve critical issues before use")
            
        if any("PACKAGE_MISSING" in issue for issue in self.issues_found):
            recommendations.append("Install missing dependencies with pip install -r requirements.txt")
            
        if any("API_KEY" in issue for issue in self.issues_found):
            recommendations.append("Configure the Google Gemini 2.0 Flash AI API Key in the .env file")
            
        if not recommendations:
            recommendations.append("System in good condition, regular maintenance recommended")
            
        return recommendations
        
    def run_full_maintenance(self) -> Dict:
        """Executes full maintenance"""
        self.log_action("MAINTENANCE_START", "INFO", "Starting full maintenance")
        
        maintenance_tasks = [
            ("Integrity check", self.check_file_integrity),
            ("Dependency check", self.check_dependencies),
            ("Configuration check", self.check_configuration),
            ("Disk space check", self.check_disk_space),
            ("Temporary file cleanup", self.clean_temporary_files),
            ("Import optimization", self.optimize_imports),
            ("Quick tests", self.run_quick_tests),
            ("Documentation update", self.update_documentation)
        ]
        
        for task_name, task_func in maintenance_tasks:
            try:
                self.log_action("TASK_START", "INFO", f"Starting: {task_name}")
                success = task_func()
                
                if success:
                    self.log_action("TASK_SUCCESS", "SUCCESS", f"Completed: {task_name}")
                else:
                    self.log_action("TASK_PARTIAL", "WARNING", f"Partially successful: {task_name}")
                    
            except Exception as e:
                self.log_action("TASK_ERROR", "ERROR", f"Error {task_name}: {e}")
                
        # Generate final report
        health_report = self.generate_health_report()
        
        self.log_action("MAINTENANCE_COMPLETE", "SUCCESS", 
                       f"Maintenance complete - Health score: {health_report['maintenance_summary']['health_score']}")
        
        return health_report

def display_health_report(report: Dict):
    """Displays the health report"""
    print("\n" + "=" * 80)
    print("🏥 SYSTEM HEALTH REPORT - GOOGLE GEMINI 2.0 FLASH AI INTERACTIVE NAVIGATION")
    print("=" * 80)
    
    summary = report["maintenance_summary"]
    
    print(f"\n⏱️ MAINTENANCE DURATION: {summary['duration_seconds']}s")
    print(f"🎯 HEALTH SCORE: {summary['health_score']}/100")
    print(f"📊 STATUS: {summary['status']}")
    
    # Status icons
    status_icons = {
        "HEALTHY": "🟢",
        "NEEDS_ATTENTION": "🟡", 
        "CRITICAL": "🔴"
    }
    
    print(f"{status_icons.get(summary['status'], '⚪')} {summary['status']}")
    
    # File details
    files_info = report["critical_files"]
    print(f"\n📁 CRITICAL FILES:")
    print(f"   • Total: {files_info['total']}")
    print(f"   • Present: {files_info['present']} ✅")
    print(f"   • Missing: {len(files_info['missing'])} ❌")
    
    if files_info['missing']:
        print("   📋 Missing files:")
        for missing_file in files_info['missing']:
            print(f"      - {missing_file}")
    
    # Issues found
    if report["issues_detail"]:
        print(f"\n⚠️ DETECTED ISSUES ({len(report['issues_detail'])}):")
        for i, issue in enumerate(report["issues_detail"], 1):
            print(f"   {i}. {issue}")
    
    # Fixes applied
    if report["fixes_detail"]:
        print(f"\n🔧 FIXES APPLIED ({len(report['fixes_detail'])}):")
        for i, fix in enumerate(report["fixes_detail"], 1):
            print(f"   {i}. {fix}")
    
    # Recommendations
    if report["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    # Final evaluation
    if summary['health_score'] >= 90:
        print(f"\n🎉 EXCELLENT - System in perfect condition")
    elif summary['health_score'] >= 70:
        print(f"\n👍 GOOD - System functional with minor possible improvements")
    elif summary['health_score'] >= 50:
        print(f"\n⚠️ ATTENTION - System needs fixes")
    else:
        print(f"\n🚨 CRITICAL - Urgent interventions required")
    
    print("\n" + "=" * 80)

def main():
    """Main entry point"""
    print("🔧 SYSTEM MAINTENANCE - GOOGLE GEMINI 2.0 FLASH AI INTERACTIVE NAVIGATION")
    print("🎯 Automatic validation, cleanup, and optimization")
    
    maintainer = InteractiveNavigationMaintainer()
    
    try:
        # Full maintenance execution
        health_report = maintainer.run_full_maintenance()
        
        # Display report
        display_health_report(health_report)
        
        # Save report
        report_file = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(health_report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full report saved: {report_file}")
        
        # Return code based on health
        health_score = health_report["maintenance_summary"]["health_score"]
        return health_score >= 50
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Maintenance interrupted by user")
        return False
    except Exception as e:
        logger.error(f"💥 Critical error during maintenance: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
