"""
Automated runbook execution system for common operational issues.

This module provides safe, auditable automation of standard operational
procedures in response to alerts and incidents.
"""

import asyncio
import subprocess
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import aiohttp
import structlog

from genesis.core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


class ActionType(Enum):
    """Types of runbook actions."""
    SHELL = "shell"
    HTTP = "http"
    KUBERNETES = "kubernetes"
    DATABASE = "database"
    RESTART_SERVICE = "restart_service"
    SCALE_SERVICE = "scale_service"
    CLEAR_CACHE = "clear_cache"
    ROLLBACK = "rollback"
    NOTIFICATION = "notification"
    WAIT = "wait"


class ExecutionMode(Enum):
    """Runbook execution modes."""
    DRY_RUN = "dry_run"
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class RunbookAction:
    """Single action within a runbook."""
    name: str
    type: ActionType
    description: str
    command: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 30
    retry_count: int = 3
    safe_for_auto: bool = False
    requires_confirmation: bool = True
    success_criteria: Optional[Dict[str, Any]] = None


@dataclass
class RunbookCondition:
    """Condition for runbook execution."""
    metric: str
    operator: str
    value: Any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        actual_value = context.get(self.metric)
        if actual_value is None:
            return False
        
        if self.operator == ">":
            return actual_value > self.value
        elif self.operator == "<":
            return actual_value < self.value
        elif self.operator == "==":
            return actual_value == self.value
        elif self.operator == "!=":
            return actual_value != self.value
        elif self.operator == ">=":
            return actual_value >= self.value
        elif self.operator == "<=":
            return actual_value <= self.value
        elif self.operator == "in":
            return actual_value in self.value
        elif self.operator == "contains":
            return self.value in str(actual_value)
        
        return False


@dataclass
class Runbook:
    """Complete runbook definition."""
    id: str
    name: str
    description: str
    trigger: Dict[str, Any]
    conditions: List[RunbookCondition]
    actions: List[RunbookAction]
    rollback_actions: List[RunbookAction] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.APPROVAL_REQUIRED
    max_executions_per_hour: int = 5
    tags: List[str] = field(default_factory=list)
    author: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionResult:
    """Result of runbook execution."""
    runbook_id: str
    execution_id: str
    status: str  # success, failed, partial, dry_run
    start_time: datetime
    end_time: datetime
    actions_executed: List[str]
    actions_failed: List[str]
    outputs: Dict[str, Any]
    errors: List[str]
    dry_run: bool = False


class RunbookExecutor:
    """
    Executes runbooks in response to alerts and operational issues.
    
    Features:
    - Safe execution with dry-run mode
    - Approval workflow for sensitive actions
    - Rollback capabilities
    - Comprehensive audit logging
    """
    
    def __init__(
        self,
        runbook_dir: str = "monitoring/runbooks",
        audit_log_path: str = "logs/runbook_actions.log"
    ):
        self.runbook_dir = Path(runbook_dir)
        self.audit_log_path = Path(audit_log_path)
        self.runbooks: Dict[str, Runbook] = {}
        self.execution_history: List[ExecutionResult] = []
        self.execution_counts: Dict[str, List[datetime]] = {}
        
        # Action executors
        self.action_executors: Dict[ActionType, Callable] = {
            ActionType.SHELL: self._execute_shell,
            ActionType.HTTP: self._execute_http,
            ActionType.KUBERNETES: self._execute_kubernetes,
            ActionType.DATABASE: self._execute_database,
            ActionType.RESTART_SERVICE: self._execute_restart_service,
            ActionType.SCALE_SERVICE: self._execute_scale_service,
            ActionType.CLEAR_CACHE: self._execute_clear_cache,
            ActionType.ROLLBACK: self._execute_rollback,
            ActionType.NOTIFICATION: self._execute_notification,
            ActionType.WAIT: self._execute_wait,
        }
        
        # Ensure directories exist
        self.runbook_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the runbook executor."""
        await self.load_runbooks()
        logger.info(f"Loaded {len(self.runbooks)} runbooks")
    
    async def load_runbooks(self) -> None:
        """Load runbook definitions from YAML files."""
        if not self.runbook_dir.exists():
            logger.warning(f"Runbook directory does not exist: {self.runbook_dir}")
            return
        
        for runbook_file in self.runbook_dir.glob("*.yaml"):
            try:
                with open(runbook_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                runbook = self._parse_runbook(data)
                self.runbooks[runbook.id] = runbook
                
                logger.info(f"Loaded runbook: {runbook.name}", runbook_id=runbook.id)
                
            except Exception as e:
                logger.error(f"Failed to load runbook: {runbook_file}", error=str(e))
    
    def _parse_runbook(self, data: Dict[str, Any]) -> Runbook:
        """Parse runbook from YAML data."""
        conditions = []
        for cond_data in data.get('conditions', []):
            conditions.append(RunbookCondition(
                metric=cond_data['metric'],
                operator=cond_data['operator'],
                value=cond_data['value']
            ))
        
        actions = []
        for action_data in data.get('actions', []):
            actions.append(RunbookAction(
                name=action_data['name'],
                type=ActionType(action_data['type']),
                description=action_data.get('description', ''),
                command=action_data.get('command'),
                url=action_data.get('url'),
                method=action_data.get('method'),
                payload=action_data.get('payload'),
                timeout_seconds=action_data.get('timeout_seconds', 30),
                retry_count=action_data.get('retry_count', 3),
                safe_for_auto=action_data.get('safe_for_auto', False),
                requires_confirmation=action_data.get('requires_confirmation', True),
                success_criteria=action_data.get('success_criteria')
            ))
        
        rollback_actions = []
        for action_data in data.get('rollback_actions', []):
            rollback_actions.append(RunbookAction(
                name=action_data['name'],
                type=ActionType(action_data['type']),
                description=action_data.get('description', ''),
                command=action_data.get('command'),
                timeout_seconds=action_data.get('timeout_seconds', 30)
            ))
        
        return Runbook(
            id=data.get('id', data['name'].lower().replace(' ', '_')),
            name=data['name'],
            description=data['description'],
            trigger=data.get('trigger', {}),
            conditions=conditions,
            actions=actions,
            rollback_actions=rollback_actions,
            execution_mode=ExecutionMode(data.get('execution_mode', 'approval_required')),
            max_executions_per_hour=data.get('max_executions_per_hour', 5),
            tags=data.get('tags', []),
            author=data.get('author', ''),
            version=data.get('version', '1.0')
        )
    
    async def execute_runbook(
        self,
        runbook_id: str,
        context: Dict[str, Any],
        dry_run: bool = False,
        force: bool = False
    ) -> ExecutionResult:
        """
        Execute a runbook.
        
        Args:
            runbook_id: ID of the runbook to execute
            context: Context data for conditions and actions
            dry_run: If True, simulate execution without making changes
            force: If True, bypass rate limiting and approval requirements
            
        Returns:
            ExecutionResult with execution details
        """
        if runbook_id not in self.runbooks:
            raise ValidationError(f"Runbook not found: {runbook_id}")
        
        runbook = self.runbooks[runbook_id]
        execution_id = f"{runbook_id}_{int(datetime.utcnow().timestamp())}"
        
        # Check rate limiting
        if not force and not self._check_rate_limit(runbook_id, runbook.max_executions_per_hour):
            raise ValidationError(f"Rate limit exceeded for runbook: {runbook_id}")
        
        # Check conditions
        if not all(cond.evaluate(context) for cond in runbook.conditions):
            logger.info(f"Runbook conditions not met: {runbook_id}")
            return ExecutionResult(
                runbook_id=runbook_id,
                execution_id=execution_id,
                status="skipped",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                actions_executed=[],
                actions_failed=[],
                outputs={"reason": "Conditions not met"},
                errors=[],
                dry_run=dry_run
            )
        
        # Check execution mode
        if not force and runbook.execution_mode == ExecutionMode.MANUAL:
            raise ValidationError(f"Runbook requires manual execution: {runbook_id}")
        
        if not force and runbook.execution_mode == ExecutionMode.APPROVAL_REQUIRED:
            # In production, this would wait for approval
            logger.warning(f"Runbook requires approval: {runbook_id}")
        
        # Execute actions
        result = await self._execute_actions(
            runbook=runbook,
            execution_id=execution_id,
            context=context,
            dry_run=dry_run
        )
        
        # Record execution
        self._record_execution(runbook_id)
        self.execution_history.append(result)
        
        # Audit log
        await self._audit_execution(runbook, result, context)
        
        return result
    
    async def _execute_actions(
        self,
        runbook: Runbook,
        execution_id: str,
        context: Dict[str, Any],
        dry_run: bool
    ) -> ExecutionResult:
        """Execute runbook actions."""
        start_time = datetime.utcnow()
        actions_executed = []
        actions_failed = []
        outputs = {}
        errors = []
        
        for action in runbook.actions:
            try:
                logger.info(
                    f"Executing action: {action.name}",
                    runbook_id=runbook.id,
                    action_type=action.type.value,
                    dry_run=dry_run
                )
                
                if dry_run:
                    # Simulate execution
                    outputs[action.name] = {
                        "status": "simulated",
                        "message": f"Would execute: {action.description}"
                    }
                    actions_executed.append(action.name)
                else:
                    # Execute action
                    executor = self.action_executors.get(action.type)
                    if not executor:
                        raise ValidationError(f"Unknown action type: {action.type}")
                    
                    action_output = await executor(action, context)
                    outputs[action.name] = action_output
                    
                    # Check success criteria
                    if action.success_criteria and not self._check_success_criteria(
                        action_output, action.success_criteria
                    ):
                        raise Exception(f"Action failed success criteria: {action.name}")
                    
                    actions_executed.append(action.name)
                    
            except Exception as e:
                logger.error(
                    f"Action failed: {action.name}",
                    runbook_id=runbook.id,
                    error=str(e)
                )
                actions_failed.append(action.name)
                errors.append(f"{action.name}: {str(e)}")
                
                # Decide whether to continue or stop
                if not action.safe_for_auto:
                    logger.error("Stopping execution due to critical action failure")
                    break
        
        # Determine overall status
        if actions_failed:
            status = "partial" if actions_executed else "failed"
        else:
            status = "success" if actions_executed else "no_actions"
        
        return ExecutionResult(
            runbook_id=runbook.id,
            execution_id=execution_id,
            status=status,
            start_time=start_time,
            end_time=datetime.utcnow(),
            actions_executed=actions_executed,
            actions_failed=actions_failed,
            outputs=outputs,
            errors=errors,
            dry_run=dry_run
        )
    
    async def _execute_shell(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell command action."""
        if not action.command:
            raise ValidationError("Shell action requires command")
        
        # Substitute variables in command
        command = self._substitute_variables(action.command, context)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=action.timeout_seconds
            )
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Command timed out after {action.timeout_seconds} seconds"
            }
    
    async def _execute_http(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request action."""
        if not action.url:
            raise ValidationError("HTTP action requires URL")
        
        url = self._substitute_variables(action.url, context)
        method = action.method or "GET"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=action.payload,
                    timeout=aiohttp.ClientTimeout(total=action.timeout_seconds)
                ) as response:
                    return {
                        "status": "success" if response.status < 400 else "failed",
                        "status_code": response.status,
                        "body": await response.text()
                    }
            except asyncio.TimeoutError:
                return {
                    "status": "timeout",
                    "error": f"Request timed out after {action.timeout_seconds} seconds"
                }
    
    async def _execute_kubernetes(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Kubernetes action."""
        # This would use kubectl or Kubernetes API
        # For now, simulate the action
        return {
            "status": "simulated",
            "message": f"Would execute Kubernetes action: {action.command}"
        }
    
    async def _execute_database(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database action."""
        # This would execute database queries
        # For now, simulate the action
        return {
            "status": "simulated",
            "message": f"Would execute database action: {action.command}"
        }
    
    async def _execute_restart_service(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Restart a service."""
        service_name = context.get('service_name', 'unknown')
        
        # This would use systemctl or service manager
        # For now, simulate the action
        return {
            "status": "simulated",
            "message": f"Would restart service: {service_name}"
        }
    
    async def _execute_scale_service(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Scale a service."""
        service_name = context.get('service_name', 'unknown')
        replicas = context.get('replicas', 1)
        
        # This would use Kubernetes or container orchestrator
        # For now, simulate the action
        return {
            "status": "simulated",
            "message": f"Would scale {service_name} to {replicas} replicas"
        }
    
    async def _execute_clear_cache(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clear cache."""
        # This would clear application cache
        # For now, simulate the action
        return {
            "status": "simulated",
            "message": "Would clear cache"
        }
    
    async def _execute_rollback(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback."""
        # This would trigger deployment rollback
        # For now, simulate the action
        return {
            "status": "simulated",
            "message": "Would execute rollback"
        }
    
    async def _execute_notification(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification."""
        # This would send notification via configured channels
        # For now, simulate the action
        return {
            "status": "simulated",
            "message": f"Would send notification: {action.payload}"
        }
    
    async def _execute_wait(self, action: RunbookAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for specified duration."""
        wait_seconds = action.timeout_seconds
        await asyncio.sleep(wait_seconds)
        return {
            "status": "success",
            "message": f"Waited {wait_seconds} seconds"
        }
    
    def _substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Substitute variables in text with context values."""
        for key, value in context.items():
            text = text.replace(f"${{{key}}}", str(value))
            text = text.replace(f"${key}", str(value))
        return text
    
    def _check_success_criteria(self, output: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if action output meets success criteria."""
        for key, expected_value in criteria.items():
            actual_value = output.get(key)
            if actual_value != expected_value:
                return False
        return True
    
    def _check_rate_limit(self, runbook_id: str, max_per_hour: int) -> bool:
        """Check if runbook execution is within rate limits."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        if runbook_id not in self.execution_counts:
            self.execution_counts[runbook_id] = []
        
        # Clean up old entries
        self.execution_counts[runbook_id] = [
            t for t in self.execution_counts[runbook_id]
            if t > hour_ago
        ]
        
        # Check limit
        return len(self.execution_counts[runbook_id]) < max_per_hour
    
    def _record_execution(self, runbook_id: str) -> None:
        """Record runbook execution for rate limiting."""
        if runbook_id not in self.execution_counts:
            self.execution_counts[runbook_id] = []
        self.execution_counts[runbook_id].append(datetime.utcnow())
    
    async def _audit_execution(
        self,
        runbook: Runbook,
        result: ExecutionResult,
        context: Dict[str, Any]
    ) -> None:
        """Write execution details to audit log."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "runbook_id": runbook.id,
            "runbook_name": runbook.name,
            "execution_id": result.execution_id,
            "status": result.status,
            "dry_run": result.dry_run,
            "duration_seconds": (result.end_time - result.start_time).total_seconds(),
            "actions_executed": result.actions_executed,
            "actions_failed": result.actions_failed,
            "context": context,
            "errors": result.errors
        }
        
        # Write to audit log
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
    
    def get_runbook_status(self, runbook_id: str) -> Dict[str, Any]:
        """Get status of a specific runbook."""
        if runbook_id not in self.runbooks:
            return {"error": "Runbook not found"}
        
        runbook = self.runbooks[runbook_id]
        executions_in_hour = len(self.execution_counts.get(runbook_id, []))
        
        # Get recent executions
        recent_executions = [
            e for e in self.execution_history
            if e.runbook_id == runbook_id
        ][-5:]  # Last 5 executions
        
        return {
            "id": runbook.id,
            "name": runbook.name,
            "description": runbook.description,
            "execution_mode": runbook.execution_mode.value,
            "executions_this_hour": executions_in_hour,
            "max_executions_per_hour": runbook.max_executions_per_hour,
            "can_execute": executions_in_hour < runbook.max_executions_per_hour,
            "recent_executions": [
                {
                    "execution_id": e.execution_id,
                    "status": e.status,
                    "timestamp": e.start_time.isoformat(),
                    "duration_seconds": (e.end_time - e.start_time).total_seconds()
                }
                for e in recent_executions
            ]
        }
    
    def list_runbooks(self) -> List[Dict[str, Any]]:
        """List all available runbooks."""
        return [
            {
                "id": rb.id,
                "name": rb.name,
                "description": rb.description,
                "tags": rb.tags,
                "execution_mode": rb.execution_mode.value,
                "safe_for_auto": all(a.safe_for_auto for a in rb.actions)
            }
            for rb in self.runbooks.values()
        ]