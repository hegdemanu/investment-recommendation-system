#!/usr/bin/env python3
import os
from pathlib import Path
import json
from typing import Dict, List, Set
import re

def get_file_type(file_path: str) -> str:
    """Determine the type of file based on extension and content."""
    ext = Path(file_path).suffix.lower()
    
    # Map extensions to file types
    type_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'React',
        '.ts': 'TypeScript',
        '.tsx': 'React TypeScript',
        '.json': 'JSON',
        '.yml': 'YAML',
        '.yaml': 'YAML',
        '.md': 'Documentation',
        '.env': 'Environment',
        '.sql': 'SQL',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.html': 'HTML',
        '.log': 'Log',
        '.ini': 'Configuration',
        '.toml': 'Configuration',
        '.cfg': 'Configuration'
    }
    
    return type_map.get(ext, 'Other')

def analyze_python_imports(content: str) -> Set[str]:
    """Extract Python imports from file content."""
    imports = set()
    import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+([^as\s]+))'
    
    for line in content.split('\n'):
        match = re.match(import_pattern, line.strip())
        if match:
            module = match.group(1) or match.group(2)
            imports.add(module.split('.')[0])
    
    return imports

def analyze_js_imports(content: str) -> Set[str]:
    """Extract JavaScript/TypeScript imports from file content."""
    imports = set()
    import_pattern = r'(?:import|require)\s*\(?[\'"]([^\'"\s]+)[\'"]'
    
    for match in re.finditer(import_pattern, content):
        module = match.group(1)
        # Get the package name (first part of the path)
        package = module.split('/')[0]
        if not package.startswith('.'):
            imports.add(package)
    
    return imports

def analyze_file(file_path: str) -> Dict:
    """Analyze a single file and return its metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                lines = len(content.split('\n'))
            except UnicodeDecodeError:
                return {
                    'path': file_path,
                    'type': get_file_type(file_path),
                    'size': os.path.getsize(file_path),
                    'imports': [],
                    'lines': 0,
                    'error': 'Binary or non-UTF-8 file'
                }
        
        file_type = get_file_type(file_path)
        imports = set()
        
        if file_type == 'Python':
            imports = analyze_python_imports(content)
        elif file_type in ['JavaScript', 'TypeScript', 'React', 'React TypeScript']:
            imports = analyze_js_imports(content)
        
        return {
            'path': file_path,
            'type': file_type,
            'size': os.path.getsize(file_path),
            'imports': sorted(list(imports)),
            'lines': lines
        }
    except Exception as e:
        return {
            'path': file_path,
            'type': get_file_type(file_path),
            'size': os.path.getsize(file_path),
            'imports': [],
            'lines': 0,
            'error': str(e)
        }

def categorize_file(file_info: Dict) -> str:
    """Categorize a file into one of the main project directories."""
    path = file_info['path']
    file_type = file_info['type']
    
    # Check existing directory first
    if path.startswith(('backend/', 'frontend/', 'trading_engine/', 'data_pipeline/', 'docs/', 'scripts/', 'archive/')):
        return path.split('/')[0]
    
    # Categorize based on file type and content
    if file_type == 'Python':
        if any(kw in path.lower() for kw in ['model', 'predict', 'train', 'ml', 'ai']):
            return 'trading_engine'
        if any(kw in path.lower() for kw in ['api', 'endpoint', 'route', 'view']):
            return 'backend'
        if any(kw in path.lower() for kw in ['data', 'pipeline', 'process']):
            return 'data_pipeline'
    elif file_type in ['JavaScript', 'TypeScript', 'React', 'React TypeScript', 'CSS', 'SCSS', 'HTML']:
        return 'frontend'
    elif file_type == 'Documentation':
        return 'docs'
    elif path.startswith('scripts/'):
        return 'scripts'
    
    return 'root'

def analyze_codebase() -> Dict:
    """Analyze the entire codebase and generate a report."""
    analysis = {
        'files': [],
        'summary': {
            'total_files': 0,
            'total_lines': 0,
            'by_type': {},
            'by_category': {},
            'dependencies': {
                'python': set(),
                'javascript': set()
            }
        }
    }
    
    # Walk through the repository
    for root, _, files in os.walk('.'):
        if any(ignore in root for ignore in ['.git', '__pycache__', 'node_modules', 'venv', '.env']):
            continue
            
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)[2:]  # Remove './'
            file_info = analyze_file(file_path)
            category = categorize_file(file_info)
            
            # Update file info with category
            file_info['category'] = category
            analysis['files'].append(file_info)
            
            # Update summary
            analysis['summary']['total_files'] += 1
            analysis['summary']['total_lines'] += file_info['lines']
            analysis['summary']['by_type'][file_info['type']] = analysis['summary']['by_type'].get(file_info['type'], 0) + 1
            analysis['summary']['by_category'][category] = analysis['summary']['by_category'].get(category, 0) + 1
            
            # Update dependencies
            if file_info['type'] == 'Python':
                analysis['summary']['dependencies']['python'].update(file_info['imports'])
            elif file_info['type'] in ['JavaScript', 'TypeScript', 'React', 'React TypeScript']:
                analysis['summary']['dependencies']['javascript'].update(file_info['imports'])
    
    # Convert sets to sorted lists for JSON serialization
    analysis['summary']['dependencies']['python'] = sorted(list(analysis['summary']['dependencies']['python']))
    analysis['summary']['dependencies']['javascript'] = sorted(list(analysis['summary']['dependencies']['javascript']))
    
    return analysis

def generate_report(analysis: Dict) -> str:
    """Generate a markdown report from the analysis."""
    report = [
        "# Codebase Analysis Report\n",
        "## Summary\n",
        f"- Total Files: {analysis['summary']['total_files']}",
        f"- Total Lines of Code: {analysis['summary']['total_lines']}\n",
        
        "## Files by Category\n",
        *[f"- {cat}: {count}" for cat, count in sorted(analysis['summary']['by_category'].items())],
        "\n## Files by Type\n",
        *[f"- {type_}: {count}" for type_, count in sorted(analysis['summary']['by_type'].items())],
        "\n## Dependencies\n",
        "### Python Dependencies\n",
        *[f"- {dep}" for dep in analysis['summary']['dependencies']['python']],
        "\n### JavaScript Dependencies\n",
        *[f"- {dep}" for dep in analysis['summary']['dependencies']['javascript']],
        "\n## Files by Category\n"
    ]
    
    # Group files by category
    files_by_category = {}
    for file in analysis['files']:
        category = file['category']
        if category not in files_by_category:
            files_by_category[category] = []
        files_by_category[category].append(file)
    
    # Add files grouped by category
    for category, files in sorted(files_by_category.items()):
        report.extend([
            f"\n### {category}\n",
            *[f"- {f['path']} ({f['type']}, {f['lines']} lines)" for f in sorted(files, key=lambda x: x['path'])]
        ])
    
    return '\n'.join(report)

def main():
    """Main function to analyze codebase and generate report."""
    print("Analyzing codebase...")
    analysis = analyze_codebase()
    
    # Save raw analysis as JSON
    with open('docs/architecture/codebase_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Generate and save report
    report = generate_report(analysis)
    with open('docs/architecture/codebase_analysis.md', 'w') as f:
        f.write(report)
    
    print("\nAnalysis complete!")
    print("- Raw analysis saved to: docs/architecture/codebase_analysis.json")
    print("- Report saved to: docs/architecture/codebase_analysis.md")

if __name__ == "__main__":
    main() 