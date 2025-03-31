"""
Background service for asynchronous report generation.
"""
import uuid
import logging
import time
import threading
from queue import Queue
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerationService:
    """Service for background report generation tasks"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one service instance"""
        if cls._instance is None:
            cls._instance = super(ReportGenerationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_workers: int = 1):
        """Initialize the report generation service"""
        if self._initialized:
            return
            
        self.task_queue = Queue()
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self.in_progress: Dict[str, datetime] = {}
        self.max_workers = max_workers
        self._stop_event = threading.Event()
        
        # Start worker threads
        self._workers = []
        for i in range(max_workers):
            worker = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name=f"report-worker-{i}"
            )
            self._workers.append(worker)
            worker.start()
            
        logger.info(f"ReportGenerationService initialized with {max_workers} workers")
        self._initialized = True

    def submit_report_task(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a report generation task
        
        Args:
            query: The query to analyze
            parameters: Optional parameters for the report generation
            
        Returns:
            task_id: Unique ID for tracking the task
        """
        if not query:
            raise ValueError("Query cannot be empty")
            
        task_id = str(uuid.uuid4())
        
        if parameters is None:
            parameters = {}
            
        # Add task to queue
        self.task_queue.put((task_id, query, parameters))
        self.in_progress[task_id] = datetime.now()
        
        logger.info(f"Submitted report task {task_id}: {query[:50]}{'...' if len(query) > 50 else ''}")
        return task_id

    def get_report_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a report generation task
        
        Args:
            task_id: The task ID to check
            
        Returns:
            status: A dictionary with the task status
        """
        if task_id in self.results:
            return {
                "status": "completed",
                "task_id": task_id,
                "result": self.results[task_id]
            }
        elif task_id in self.errors:
            return {
                "status": "failed",
                "task_id": task_id,
                "error": self.errors[task_id]
            }
        elif task_id in self.in_progress:
            return {
                "status": "in_progress",
                "task_id": task_id,
                "submitted_at": self.in_progress[task_id].isoformat()
            }
        else:
            return {
                "status": "unknown",
                "task_id": task_id
            }

    def get_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all tasks with their statuses
        
        Returns:
            tasks: Dictionary with completed, failed, and in_progress tasks
        """
        completed = []
        for task_id, result in self.results.items():
            completed.append({
                "task_id": task_id,
                "status": "completed",
                "title": result.get("title", "Untitled Report"),
                "generated_at": result.get("generated_at", "Unknown")
            })
            
        failed = []
        for task_id, error in self.errors.items():
            failed.append({
                "task_id": task_id,
                "status": "failed",
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
            
        in_progress = []
        for task_id, timestamp in self.in_progress.items():
            if task_id not in self.results and task_id not in self.errors:
                in_progress.append({
                    "task_id": task_id,
                    "status": "in_progress",
                    "submitted_at": timestamp.isoformat()
                })
                
        return {
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress
        }

    def _process_queue(self):
        """Process tasks from the queue (runs in worker thread)"""
        while not self._stop_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    task_id, query, parameters = self.task_queue.get(timeout=1)
                except Exception:
                    # No task available, continue polling
                    continue
                    
                logger.info(f"Processing report task {task_id}: {query[:30]}...")
                
                try:
                    # Import here to avoid circular imports
                    from app.core.rag_system import EnhancedRAGSystem
                    
                    # Initialize RAG system
                    rag = EnhancedRAGSystem()
                    
                    # Generate report
                    report = rag.generate_analysis_report(query)
                    
                    # Store result
                    self.results[task_id] = report
                    
                    logger.info(f"Completed report task {task_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing report task {task_id}: {str(e)}")
                    self.errors[task_id] = str(e)
                    
                finally:
                    # Remove from in_progress regardless of success/failure
                    if task_id in self.in_progress:
                        del self.in_progress[task_id]
                    
                    # Mark task as done
                    self.task_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Error in report worker: {str(e)}")
                # Sleep briefly to avoid tight loop on persistent errors
                time.sleep(0.1)
                
    def stop(self):
        """Stop the service and all worker threads"""
        logger.info("Stopping ReportGenerationService...")
        self._stop_event.set()
        
        # Wait for all workers to finish
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=2)
                
        logger.info("ReportGenerationService stopped")
        
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        self.stop()

    def clean_old_results(self, max_age_hours: int = 24):
        """
        Clean up old results to prevent memory leaks
        
        Args:
            max_age_hours: Maximum age of results to keep in memory
        """
        cutoff = datetime.now()
        # This is a placeholder - we'd need to store timestamps for each result
        # to implement this properly.
        pass

    def test_report_generation_flow(self):
        test_query = "Test query"
        task_id = self.submit_report_task(test_query)
        while task_id not in self.results:
            pass
        report = self.results[task_id]
        assert report is not None

    def benchmark_analysis_quality(self, previous_version):
        test_queries = ["Query 1", "Query 2", "Query 3"]
        previous_reports = [previous_version.generate_analysis_report(query) for query in test_queries]
        current_reports = [self.generate_analysis_report(query) for query in test_queries]
        # Compare previous_reports and current_reports

    def stress_test_real_time_generation(self, num_queries):
        task_ids = [self.submit_report_task(f"Query {i}") for i in range(num_queries)]
        while any(task_id not in self.results for task_id in task_ids):
            pass
        reports = [self.results[task_id] for task_id in task_ids]
        assert all(report is not None for report in reports)