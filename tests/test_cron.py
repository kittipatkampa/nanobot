"""Tests for the cron service."""

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest

from nanobot.cron.service import CronService, _compute_next_run, _now_ms
from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule


class TestComputeNextRun:
    """Test _compute_next_run function."""

    def test_at_schedule_future(self):
        """Test 'at' schedule with future timestamp."""
        now = _now_ms()
        future_ms = now + 60000  # 1 minute in future
        schedule = CronSchedule(kind="at", at_ms=future_ms)
        result = _compute_next_run(schedule, now)
        assert result == future_ms

    def test_at_schedule_past(self):
        """Test 'at' schedule with past timestamp returns None."""
        now = _now_ms()
        past_ms = now - 60000  # 1 minute in past
        schedule = CronSchedule(kind="at", at_ms=past_ms)
        result = _compute_next_run(schedule, now)
        assert result is None

    def test_every_schedule(self):
        """Test 'every' schedule computes correct next run."""
        now = _now_ms()
        every_ms = 30000  # 30 seconds
        schedule = CronSchedule(kind="every", every_ms=every_ms)
        result = _compute_next_run(schedule, now)
        assert result == now + every_ms

    def test_every_schedule_zero(self):
        """Test 'every' schedule with zero interval returns None."""
        schedule = CronSchedule(kind="every", every_ms=0)
        result = _compute_next_run(schedule, _now_ms())
        assert result is None

    def test_cron_schedule(self):
        """Test cron expression schedule."""
        schedule = CronSchedule(kind="cron", expr="* * * * *")  # every minute
        result = _compute_next_run(schedule, _now_ms())
        assert result is not None
        # Should be within the next 60 seconds (roughly)
        assert result > _now_ms()
        assert result <= _now_ms() + 61000

    def test_invalid_cron_expression(self):
        """Test invalid cron expression returns None."""
        schedule = CronSchedule(kind="cron", expr="invalid")
        result = _compute_next_run(schedule, _now_ms())
        assert result is None


class TestCronService:
    """Test CronService class."""

    @pytest.fixture
    def temp_store_path(self):
        """Create a temporary file path for the cron store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "cron" / "jobs.json"

    @pytest.fixture
    def service(self, temp_store_path):
        """Create a CronService instance."""
        return CronService(temp_store_path)

    def test_add_job_every(self, service):
        """Test adding a job with 'every' schedule."""
        job = service.add_job(
            name="Test Every",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test message",
        )
        assert job.id is not None
        assert job.name == "Test Every"
        assert job.enabled is True
        assert job.schedule.kind == "every"
        assert job.schedule.every_ms == 60000
        assert job.payload.message == "Test message"
        assert job.state.next_run_at_ms is not None

    def test_add_job_cron(self, service):
        """Test adding a job with 'cron' schedule."""
        job = service.add_job(
            name="Test Cron",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *"),
            message="Daily task",
        )
        assert job.id is not None
        assert job.schedule.kind == "cron"
        assert job.schedule.expr == "0 9 * * *"

    def test_add_job_at(self, service):
        """Test adding a job with 'at' schedule."""
        future_ms = _now_ms() + 60000
        job = service.add_job(
            name="Test At",
            schedule=CronSchedule(kind="at", at_ms=future_ms),
            message="One-time task",
        )
        assert job.schedule.kind == "at"
        assert job.state.next_run_at_ms == future_ms

    def test_list_jobs(self, service):
        """Test listing jobs."""
        # Add some jobs
        service.add_job(
            name="Job 1",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test 1",
        )
        service.add_job(
            name="Job 2",
            schedule=CronSchedule(kind="every", every_ms=30000),
            message="Test 2",
        )

        jobs = service.list_jobs()
        assert len(jobs) == 2
        # Should be sorted by next_run_at_ms
        assert jobs[0].name == "Job 2"  # 30s runs sooner than 60s

    def test_remove_job(self, service):
        """Test removing a job."""
        job = service.add_job(
            name="To Remove",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test",
        )
        assert len(service.list_jobs()) == 1

        result = service.remove_job(job.id)
        assert result is True
        assert len(service.list_jobs()) == 0

    def test_remove_nonexistent_job(self, service):
        """Test removing a job that doesn't exist."""
        result = service.remove_job("nonexistent")
        assert result is False

    def test_enable_disable_job(self, service):
        """Test enabling and disabling a job."""
        job = service.add_job(
            name="Toggle",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test",
        )
        assert job.enabled is True

        # Disable
        updated = service.enable_job(job.id, enabled=False)
        assert updated is not None
        assert updated.enabled is False
        assert updated.state.next_run_at_ms is None

        # Re-enable
        updated = service.enable_job(job.id, enabled=True)
        assert updated.enabled is True
        assert updated.state.next_run_at_ms is not None

    def test_persistence(self, temp_store_path):
        """Test that jobs persist to disk."""
        # Create service and add job
        service1 = CronService(temp_store_path)
        job = service1.add_job(
            name="Persistent",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test persistence",
        )

        # Create new service instance with same path
        service2 = CronService(temp_store_path)
        jobs = service2.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == job.id
        assert jobs[0].name == "Persistent"

    def test_status(self, service):
        """Test status method."""
        status = service.status()
        assert "enabled" in status
        assert "jobs" in status
        assert "next_wake_at_ms" in status
        assert status["jobs"] == 0

        service.add_job(
            name="Status Test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test",
        )
        status = service.status()
        assert status["jobs"] == 1


class TestCronServiceAsync:
    """Test async methods of CronService."""

    @pytest.fixture
    def temp_store_path(self):
        """Create a temporary file path for the cron store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "cron" / "jobs.json"

    @pytest.mark.asyncio
    async def test_start_stop(self, temp_store_path):
        """Test starting and stopping the service."""
        service = CronService(temp_store_path)
        await service.start()
        assert service._running is True

        service.stop()
        assert service._running is False

    @pytest.mark.asyncio
    async def test_run_job_manually(self, temp_store_path):
        """Test manually running a job."""
        executed = []

        async def on_job(job):
            executed.append(job.id)
            return "OK"

        service = CronService(temp_store_path, on_job=on_job)
        job = service.add_job(
            name="Manual Run",
            schedule=CronSchedule(kind="every", every_ms=999999999),  # Far future
            message="Test",
        )

        result = await service.run_job(job.id)
        assert result is True
        assert job.id in executed

    @pytest.mark.asyncio
    async def test_job_execution_callback(self, temp_store_path):
        """Test that job callback is called on execution."""
        results = []

        async def on_job(job):
            results.append({"id": job.id, "message": job.payload.message})
            return "Executed"

        service = CronService(temp_store_path, on_job=on_job)

        # Add job that runs very soon (10ms)
        job = service.add_job(
            name="Quick Job",
            schedule=CronSchedule(kind="every", every_ms=50),
            message="Quick test",
        )

        await service.start()
        # Wait for job to execute
        await asyncio.sleep(0.2)
        service.stop()

        assert len(results) >= 1
        assert results[0]["id"] == job.id
        assert results[0]["message"] == "Quick test"


class TestCronJobState:
    """Test job state tracking."""

    @pytest.fixture
    def temp_store_path(self):
        """Create a temporary file path for the cron store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "cron" / "jobs.json"

    @pytest.mark.asyncio
    async def test_job_state_after_execution(self, temp_store_path):
        """Test that job state is updated after execution."""
        async def on_job(job):
            return "OK"

        service = CronService(temp_store_path, on_job=on_job)
        job = service.add_job(
            name="State Test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test",
        )

        assert job.state.last_run_at_ms is None
        assert job.state.last_status is None

        await service.run_job(job.id)

        # Reload job to get updated state
        jobs = service.list_jobs()
        updated_job = next(j for j in jobs if j.id == job.id)
        assert updated_job.state.last_run_at_ms is not None
        assert updated_job.state.last_status == "ok"

    @pytest.mark.asyncio
    async def test_job_state_after_error(self, temp_store_path):
        """Test that job state captures errors."""
        async def on_job(job):
            raise ValueError("Test error")

        service = CronService(temp_store_path, on_job=on_job)
        job = service.add_job(
            name="Error Test",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="Test",
        )

        await service.run_job(job.id)

        jobs = service.list_jobs()
        updated_job = next(j for j in jobs if j.id == job.id)
        assert updated_job.state.last_status == "error"
        assert "Test error" in updated_job.state.last_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
