import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

TASKS_FILE = Path("tasks_registry.json")
MAX_CONCURRENT = 3


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EARLY_STOPPED = "EARLY_STOPPED"


class TaskManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._tasks: dict[str, dict] = {}
        self._results: dict[str, object] = {}
        self._progress_buffers: dict[str, list] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._sem = threading.Semaphore(MAX_CONCURRENT)
        self._load()

    # ── Persistence ──

    def _load(self):
        if not TASKS_FILE.exists() or TASKS_FILE.stat().st_size == 0:
            return
        try:
            with open(TASKS_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            TASKS_FILE.unlink(missing_ok=True)
            return
        for t in data:
            tid = t.pop("id")
            if t["status"] == TaskStatus.RUNNING.value:
                t["status"] = TaskStatus.FAILED.value
                t["error"] = "服务重启，训练中断"
            self._tasks[tid] = t
        self._flush()

    def _flush(self):
        data = []
        for tid, t in self._tasks.items():
            entry = {"id": tid, **t}
            entry.pop("result", None)
            entry.pop("params", None)
            data.append(entry)
        tmp = TASKS_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except TypeError:
            for entry in data:
                entry.pop("result_display", None)
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(TASKS_FILE)

    # ── Task CRUD ──

    def list_tasks(self) -> list[dict]:
        items = sorted(self._tasks.items(), key=lambda x: x[1]["created_at"], reverse=True)
        for tid, t in items:
            t["_id"] = tid
        return [t for _, t in items]

    def get_task(self, task_id: str) -> dict | None:
        return self._tasks.get(task_id)

    def get_result(self, task_id: str):
        result = self._results.get(task_id)
        if result is not None:
            return result
        task = self._tasks.get(task_id)
        if task is not None:
            return task.get("result_display")
        return None

    @staticmethod
    def _extract_display_data(result: dict) -> dict:
        display = {}
        for key, val in result.items():
            if key == "agent":
                continue
            if isinstance(val, pd.Timestamp):
                display[key] = str(val)
            elif isinstance(val, pd.Series):
                records = {str(k): v for k, v in val.to_dict().items()}
                display[key] = records
            elif isinstance(val, pd.DataFrame):
                records = val.to_dict("records")
                for rec in records:
                    for k, v in rec.items():
                        if isinstance(v, pd.Timestamp):
                            rec[k] = str(v)
                display[key] = records
            elif isinstance(val, pd.DatetimeIndex):
                display[key] = [str(d) for d in val]
            elif isinstance(val, np.ndarray):
                display[key] = val.tolist()
            elif isinstance(val, (np.integer, np.floating, np.bool_)):
                display[key] = val.item()
            elif isinstance(val, dict):
                display[key] = TaskManager._extract_display_data(val)
            elif isinstance(val, (list, tuple)):
                display[key] = TaskManager._extract_display_list(val)
            else:
                display[key] = val
        return display

    @staticmethod
    def _extract_display_list(items: list | tuple) -> list:
        out = []
        for item in items:
            if isinstance(item, pd.Timestamp):
                out.append(str(item))
            elif isinstance(item, (np.integer, np.floating, np.bool_)):
                out.append(item.item())
            elif isinstance(item, dict):
                out.append(TaskManager._extract_display_data(item))
            elif isinstance(item, (list, tuple)):
                out.append(TaskManager._extract_display_list(item))
            else:
                out.append(item)
        return out

    @staticmethod
    def _extract_params_safe(params: dict) -> dict:
        safe = {}
        for k, v in params.items():
            if isinstance(v, pd.DataFrame):
                safe[k] = {"_type": "DataFrame", "rows": len(v), "columns": list(v.columns)}
            elif isinstance(v, pd.DatetimeIndex):
                safe[k] = [str(d) for d in v]
            elif isinstance(v, pd.Timestamp):
                safe[k] = str(v)
            elif isinstance(v, pd.Series):
                safe[k] = {str(i): val for i, val in v.items()}
            elif isinstance(v, np.ndarray):
                safe[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating, np.bool_)):
                safe[k] = v.item()
            elif isinstance(v, dict):
                safe[k] = TaskManager._extract_params_safe(v)
            elif isinstance(v, (list, tuple)):
                safe[k] = TaskManager._extract_display_list(v)
            else:
                safe[k] = v
        return safe

    def submit(self, task_type: str, params: dict, target_fn, args: tuple = ()) -> str:
        tid = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._tasks[tid] = {
            "type": task_type,
            "params": params,
            "params_display": self._extract_params_safe(params),
            "status": TaskStatus.PENDING.value,
            "progress": 0.0,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }
        self._cancel_events[tid] = threading.Event()
        self._flush()

        def _run():
            self._tasks[tid]["status"] = TaskStatus.RUNNING.value
            self._tasks[tid]["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._flush()
            try:
                def cancel_check():
                    return self._cancel_events[tid].is_set()
                result = target_fn(*args, task_id=tid, cancel_check=cancel_check)
                if self._cancel_events[tid].is_set():
                    if result is not None:
                        self._tasks[tid]["status"] = TaskStatus.EARLY_STOPPED.value
                        self._tasks[tid]["result_display"] = self._extract_display_data(result)
                        if tid in self._progress_buffers:
                            self._tasks[tid]["_progress_data"] = list(self._progress_buffers[tid])
                        self._results[tid] = result
                    else:
                        self._tasks[tid]["status"] = TaskStatus.CANCELLED.value
                else:
                    self._tasks[tid]["status"] = TaskStatus.COMPLETED.value
                    self._tasks[tid]["result_display"] = self._extract_display_data(result)
                    if tid in self._progress_buffers:
                        self._tasks[tid]["_progress_data"] = list(self._progress_buffers[tid])
                    self._results[tid] = result
            except Exception as e:
                self._tasks[tid]["status"] = TaskStatus.FAILED.value
                self._tasks[tid]["error"] = str(e)
            finally:
                self._tasks[tid]["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._sem.release()
                self._flush()

        self._sem.acquire()
        t = threading.Thread(target=_run, daemon=True)
        self._threads[tid] = t
        t.start()
        return tid

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task["status"] == TaskStatus.PENDING.value:
            self._cancel_events[task_id].set()
            task["status"] = TaskStatus.CANCELLED.value
            self._flush()
            return True
        if task["status"] == TaskStatus.RUNNING.value:
            self._cancel_events[task_id].set()
            return True
        return False

    def get_progress_data(self, task_id: str) -> list | None:
        return self._progress_buffers.get(task_id)

    def update_progress(self, task_id: str, progress: float, progress_data: tuple | None = None):
        task = self._tasks.get(task_id)
        if task:
            task["progress"] = round(progress, 4)
        if progress_data is not None:
            if task_id not in self._progress_buffers:
                self._progress_buffers[task_id] = []
            self._progress_buffers[task_id].append(progress_data)
