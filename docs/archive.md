# 📂 Archive Documentation

This document maintains a log of archived code from the Investment Recommendation System repository following the archiving guidelines.

## Archive Log Format

Each entry in this log follows the standard format:

```
## [Archived - YYYY-MM-DD]
### Module: <module_name>
- Version: <version_number>
- Reason: <brief reason for archiving>
- Reference: <related PR or task>
```

For code that is restored from the archive, the following format is used:

```
## [Restored - YYYY-MM-DD]
### Module: <module_name>
- Version: <version_number>
- Reason: <reason for restoration>
- Reference: <related task/PR>
```

## Archive Entries

## [Archived - 2024-05-28]
### Module: frontend
- Version: v0.1.0
- Reason: Transitioning to Next.js frontend with improved architecture
- Reference: Issue #1 - Frontend Modernization

## [Archived - 2025-03-28]
### Module: Monorepo Migration
- Version: 2.0.0
- Reason: Restructuring into a monorepo architecture
- Reference: Repository modernization

### Changes Made:
1. Reorganized into monorepo structure with apps and packages
2. Created shared UI and common packages
3. Set up Turborepo for build orchestration
4. Migrated existing components to new structure

### New Structure:
```
/investment-recommendation-system
├── apps/
│   ├── web/           # Main web application (Next.js)
│   └── admin/         # Admin dashboard
├── packages/
│   ├── ui/            # Shared UI components
│   ├── common/        # Shared utilities and types
│   ├── ml-models/     # ML service and models
│   └── data-pipeline/ # Data processing and analysis
├── docs/             # Documentation
└── scripts/          # Utility scripts
```

### Archived Components:
- Previous frontend structure -> apps/web
- API gateway -> apps/admin
- ML service -> packages/ml-models
- Trading engine -> packages/data-pipeline

## 🔄 Version Control
**Version:** 1.0.0
**Last Updated:** 2024-05-28 