## 📄 **`rules.md` - Master Guidelines for Investment Recommendation System**  

---

## 📚 **1. Objective and Scope**  
This document outlines all the core rules and processes for maintaining the Investment Recommendation System repo, including:  
- Project structure  
- Code archiving and versioning  
- Feature planning and phase-wise implementation  
- Documentation and change management  
- Security, compliance, and quality control  

---

## 🗂️ **2. Project Structure and Repo Organization**  

### 📂 **2.1 Root Folder Structure**  
```
/investment-recommendation-system
├── /archive
├── /backend
├── /frontend
├── /trading_engine
├── /data_pipeline
├── /docs
└── /scripts
```

### 📂 **2.2 Directory Guidelines**  
- **/archive:** Stores deprecated/archived code with versioning and logs.  
- **/backend:** FastAPI (Python) backend code, APIs, and endpoints.  
- **/frontend:** Next.js (TypeScript) frontend with Tailwind CSS.  
- **/trading_engine:** AI models, order execution logic, and trading strategies.  
- **/data_pipeline:** Data ingestion, preprocessing, and API integrations.  
- **/docs:** All documentation (`rules.md`, `CHANGELOG.md`, etc.).  
- **/scripts:** CI/CD, automation, and DevOps-related scripts.  

---

## 🎯 **3. Feature Planning and Phase-Wise Implementation**  

### 📝 **3.1 Phase Planning Rules**  
- **New Ideas/Features:**  
   - Must be added to `ideas.md` with phase assignment and priority.  
   - System automatically decides the correct phase to place the idea.  

- **Phase Assignment Logic:**  
   ```
   Phase 1 → Core Infrastructure & APIs  
   Phase 2 → Model Training & Backtesting  
   Phase 3 → Frontend, Dashboard & UI  
   Phase 4 → DevOps, CI/CD & Security  
   Phase 5 → Performance Optimization & Scaling  
   Phase 6 → Monetization & Compliance  
   ```

---

## 📚 **4. Versioning and Code Archiving Guidelines**  

### 📂 **4.1 Archiving Old Code**  
- **Location:** All archived code resides in `/archive` at the root level.
- **Folder Structure:**  
   ```
   /archive/<module>/<version>_<YYYY-MM-DD>/
   ```

---

## 🔥 **5. Documentation and Version Control Guidelines**  

### 📖 **5.1 Master File Maintenance**  
- **`master.md`:**  
   - Maintains all current requirements, tech stacks, and software in use.  
   - Must be updated with every major decision or version change.  

- **`CHANGELOG.md`:**  
   - Tracks all changes, including new features, bug fixes, and version updates.  
   - Follow Semantic Versioning (`vX.X.X` format).  

---

## 🔥 **Version Control**  
**Version:** 0.1.0
**Last Updated:** 2024-11-09  

--- 