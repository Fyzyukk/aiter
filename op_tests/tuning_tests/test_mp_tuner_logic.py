# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for mp_tuner polling loop logic.

Simulates async_result behavior without GPU/multiprocessing to verify:
1. consecutive_timeouts tracks correctly and resets on success
2. half-GPU threshold triggers break at the right time
3. KeyError tasks stay in remaining_tasks and get retried after root-cause restart

Run: python3 -m unittest op_tests.test_mp_tuner_logic -v
"""

import ast
import importlib
import multiprocessing as mp
import time
import unittest
from multiprocessing import TimeoutError as MPTimeoutError
from pathlib import Path
from types import SimpleNamespace


def _wait_for_release(release, value):
    release.wait(timeout=5)
    return value


class FakeAsyncResult:
    """Simulates multiprocessing.AsyncResult for testing polling logic."""

    def __init__(self, behavior, value=None):
        """
        behavior: "ok", "timeout_pending", "timeout_expired", "keyerror", "accelerator"
        value: return value for "ok"
        """
        self.behavior = behavior
        self.value = value

    def get(self, timeout=10):
        if self.behavior == "ok":
            return self.value
        elif self.behavior in ("timeout_pending", "timeout_expired"):
            raise MPTimeoutError("timeout")
        elif self.behavior == "keyerror":
            raise KeyError("12345")
        elif self.behavior == "accelerator":
            raise type("AcceleratorError", (Exception,), {})("GPU fault")


def simulate_poll_round(remaining_tasks, task_start_times, mp_num, timeout):
    """
    Simulate one round of the mp_tuner polling loop.
    Returns (completed, dummy_failed, pool_restart_needed, broke_early)
    """
    completed_this_round = []
    dummy_failed_tasks = []
    consecutive_timeouts = 0
    half_gpu = max(1, (mp_num + 1) // 2)
    pool_restart_needed = False
    broke_early = False

    for k, async_result in remaining_tasks:
        try:
            if timeout is not None:
                elapsed = time.time() - task_start_times[k]
                remaining_time = timeout - elapsed
                actual_timeout = max(1, min(10, remaining_time))
            else:
                actual_timeout = 10

            async_result.get(timeout=actual_timeout)
            completed_this_round.append((k, async_result))
            consecutive_timeouts = 0

        except MPTimeoutError:
            if timeout is not None:
                elapsed = time.time() - task_start_times[k]
                if elapsed > timeout:
                    consecutive_timeouts += 1
                    completed_this_round.append((k, async_result))
                    pool_restart_needed = True

                    if consecutive_timeouts >= half_gpu:
                        broke_early = True
                        break
                else:
                    consecutive_timeouts = 0

        except Exception as e:  # noqa: BLE001
            error_type = type(e).__name__
            is_mapping_error = error_type == "KeyError"

            if is_mapping_error:
                dummy_failed_tasks.append((k, "mapping error"))
            elif error_type == "AcceleratorError":
                completed_this_round.append((k, async_result))
                pool_restart_needed = True
                broke_early = True
                break
            else:
                completed_this_round.append((k, async_result))

    return completed_this_round, dummy_failed_tasks, pool_restart_needed, broke_early


class TestConsecutiveTimeouts(unittest.TestCase):

    def test_single_timeout_no_break_8gpu(self):
        """1 stuck GPU out of 8: should NOT break early."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("ok", [("info", 2.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, _dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Should NOT break early with interleaved success")
        self.assertTrue(restart, "Should still need restart (at least 1 timeout)")
        self.assertEqual(len(completed), 4, "All tasks should be processed")

    def test_half_gpu_consecutive_triggers_break(self):
        """4 consecutive timeouts with 8 GPUs (half=4): should break."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_expired")),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("timeout_expired")),
            (4, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, _dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "Should break after 4 consecutive timeouts (half of 8)")
        self.assertTrue(restart)
        self.assertEqual(len(completed), 4, "Task 4 not polled due to break")

    def test_success_resets_consecutive(self):
        """Success in between resets counter: 3 timeouts, 1 ok, 3 timeouts != break for 8 GPU."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_expired")),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (4, FakeAsyncResult("timeout_expired")),
            (5, FakeAsyncResult("timeout_expired")),
            (6, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, _dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Should NOT break: success at task 3 resets counter")
        self.assertTrue(restart, "Still need restart from timeouts")
        self.assertEqual(len(completed), 7)

    def test_2gpu_half_is_1(self):
        """2 GPUs: half=1, single consecutive timeout triggers break."""
        mp_num = 2
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, _dummy, _restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "2 GPUs: half=1, first timeout should break")
        self.assertEqual(len(completed), 1)

    def test_pending_timeout_resets_consecutive(self):
        """Task not yet expired (still running) resets consecutive count."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_pending")),
            (2, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {
            0: now - 200,
            1: now,
            2: now - 200,
        }

        completed, _dummy, _restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Pending task resets consecutive, so no break")
        self.assertEqual(len(completed), 2)


class TestKeyErrorHandling(unittest.TestCase):

    def test_keyerror_stays_in_remaining(self):
        """KeyError tasks should NOT be in completed_this_round."""
        mp_num = 4
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, _broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task should NOT be completed")
        self.assertIn(1, completed_ids, "OK task should be completed")
        self.assertEqual(len(dummy), 1, "KeyError task should be in dummy_failed")
        self.assertFalse(restart, "KeyError alone should NOT trigger restart")

    def test_keyerror_with_timeout_gets_resubmitted(self):
        """KeyError tasks wait for root-cause timeout to trigger restart."""
        mp_num = 2
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, _dummy, restart, _broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task stays for resubmit")
        self.assertIn(1, completed_ids, "Root-cause timeout is completed")
        self.assertTrue(restart, "Timeout should trigger restart")

        new_remaining = [(k, ar) for k, ar in remaining if k not in completed_ids]
        self.assertEqual(len(new_remaining), 1)
        self.assertEqual(
            new_remaining[0][0], 0, "Only KeyError task remains for resubmit"
        )

    def test_keyerror_no_restart_without_root_cause(self):
        """If only KeyError tasks remain, no restart, they keep polling."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("keyerror")),
        ]
        start_times = {k: now for k, _ in remaining}

        completed, dummy, restart, _broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(restart, "No restart without root cause")
        self.assertEqual(len(completed), 0, "Nothing completed")
        self.assertEqual(len(dummy), 2, "Both are mapping errors")


class TestAcceleratorError(unittest.TestCase):

    def test_accelerator_breaks_immediately(self):
        """AcceleratorError should break immediately and trigger restart."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (1, FakeAsyncResult("accelerator")),
            (2, FakeAsyncResult("ok", [("info", 2.0, 0.0)])),
        ]
        start_times = {k: now for k, _ in remaining}

        completed, _dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "AcceleratorError should break")
        self.assertTrue(restart, "AcceleratorError should trigger restart")
        completed_ids = {k for k, _ in completed}
        self.assertIn(0, completed_ids)
        self.assertIn(1, completed_ids)
        self.assertNotIn(2, completed_ids, "Task 2 not reached due to break")


class TestTaskExecutionTiming(unittest.TestCase):

    def test_queued_task_has_no_elapsed_execution_time(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        elapsed_since_start = getattr(tuner, "_elapsed_since_task_start", None)

        self.assertIsNotNone(
            elapsed_since_start,
            "mp_tuner must calculate timeout from the worker execution start",
        )
        self.assertIsNone(elapsed_since_start([0.0], 0, now=100.0))
        self.assertEqual(elapsed_since_start([55.0], 0, now=100.0), 45.0)

    def test_worker_records_start_only_when_task_leaves_queue(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        init_start_times = getattr(tuner, "_init_task_start_times", None)
        run_with_tracking = getattr(tuner, "_run_with_start_tracking", None)

        self.assertIsNotNone(init_start_times)
        self.assertIsNotNone(run_with_tracking)

        ctx = mp.get_context("spawn")
        start_times = ctx.RawArray("d", 2)
        manager = ctx.Manager()
        release = manager.Event()
        pool = ctx.Pool(1, initializer=init_start_times, initargs=(start_times,))
        try:
            first = pool.apply_async(
                run_with_tracking, (0, _wait_for_release, (release, "first"))
            )
            second = pool.apply_async(
                run_with_tracking, (1, _wait_for_release, (release, "second"))
            )

            deadline = time.monotonic() + 5
            while start_times[0] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertGreater(start_times[0], 0)
            self.assertEqual(
                start_times[1],
                0,
                "Queued task must not get a start timestamp",
            )

            release.set()
            self.assertEqual(first.get(timeout=5), "first")
            self.assertEqual(second.get(timeout=5), "second")
            self.assertGreater(start_times[1], 0)
        finally:
            release.set()
            pool.terminate()
            pool.join()
            manager.shutdown()


class TestTaskStartTimeReset(unittest.TestCase):

    def test_reset_clears_only_the_given_slots(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        reset_start_times = getattr(tuner, "_reset_task_start_times", None)

        self.assertIsNotNone(
            reset_start_times,
            "submitting a task must clear its start-time slot, otherwise a "
            "resubmitted task is judged against the previous attempt's timestamp",
        )
        slots = [11.0, 22.0, 33.0]
        reset_start_times(slots, [0, 2])
        self.assertEqual(list(slots), [0, 22.0, 0])


class TestWorkerErrorRatio(unittest.TestCase):

    def test_nonfinite_error_ratio_is_rejected(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        merge_error_ratio = getattr(tuner, "_merge_error_ratio", None)

        self.assertIsNotNone(
            merge_error_ratio,
            "worker must reject non-finite comparator error ratios",
        )
        for observed in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(observed=observed):
                self.assertEqual(merge_error_ratio(0.0, observed), 1.0)

    def test_finite_error_ratio_keeps_maximum(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        merge_error_ratio = getattr(tuner, "_merge_error_ratio", None)

        self.assertIsNotNone(merge_error_ratio)
        self.assertEqual(merge_error_ratio(0.1, 0.2), 0.2)
        self.assertEqual(merge_error_ratio(0.2, 0.1), 0.2)


class TestWorkGroupReferenceCache(unittest.TestCase):
    """Exercise the real work_group with CPU stubs, without importing torch/aiter."""

    def setUp(self):
        source_path = Path(__file__).resolve().parents[2] / "aiter/utility/mp_tuner.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        work_group = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "work_group"
        )
        self.generated = []
        self.references = []
        namespace = {
            "mp": SimpleNamespace(current_process=lambda: SimpleNamespace(pid=1)),
            "torch": SimpleNamespace(
                Tensor=type("FakeTensor", (), {}),
                device=lambda name: name,
                cuda=SimpleNamespace(
                    set_device=lambda device: None, synchronize=lambda: None
                ),
            ),
            "worker": lambda gpu_id, info, func, args, kwargs, ref, *rest: (args[0], ref),
        }
        exec(
            compile(
                ast.Module(body=[work_group], type_ignores=[]), str(source_path), "exec"
            ),
            namespace,
        )
        self.work_group = namespace["work_group"]

    def generator(self, label):
        def generate(value, *, device):
            self.generated.append((label, value))
            return {"x": (label, value)}

        return generate

    def reference(self, label):
        def reference(value, offset=0, *, factor=1):
            result = (label, value, offset, factor)
            self.references.append(result)
            return result

        return reference

    @staticmethod
    def task(generator, value, reference, offset=0, *, factor=1, explicit_ref=None):
        return (
            (("same-shape",),),
            generator,
            (value,),
            None,
            (("x",),),
            {},
            reference,
            (("x",), offset),
            {"factor": factor},
            explicit_ref,
        )

    def run_tasks(self, tasks, *, fast_mode=False):
        return self.work_group({1: 0}, fast_mode, 0.0, (len(tasks), ()), tasks)

    def test_generator_change_rebuilds_data_and_reference(self):
        fp32, e8m0 = self.generator("fp32"), self.generator("e8m0")
        reference = self.reference("reference")
        tasks = [
            self.task(gen, 1, reference) for gen in (fp32, fp32, e8m0, e8m0, fp32)
        ]
        results = self.run_tasks(tasks)
        for data, ref in results:
            self.assertEqual(ref, ("reference", data, 0, 1))
        self.assertEqual(self.generated, [("fp32", 1), ("e8m0", 1), ("fp32", 1)])
        self.assertEqual(len(self.references), 3)

    def test_generator_arguments_change_rebuilds_reference(self):
        generate, reference = self.generator("data"), self.reference("reference")
        results = self.run_tasks(
            [self.task(generate, value, reference) for value in (1, 2, 2)]
        )
        for data, ref in results:
            self.assertEqual(ref, ("reference", data, 0, 1))
        self.assertEqual(self.generated, [("data", 1), ("data", 2)])
        self.assertEqual(len(self.references), 2)

    def test_reference_function_arguments_and_keywords_invalidate_cache(self):
        generate = self.generator("data")
        first, second = self.reference("first"), self.reference("second")
        for fast_mode in (False, True):
            with self.subTest(fast_mode=fast_mode):
                self.generated.clear()
                self.references.clear()
                results = self.run_tasks(
                    [
                        self.task(generate, 1, first),
                        self.task(generate, 1, first),
                        self.task(generate, 1, first, 2),
                        self.task(generate, 1, first, 2, factor=3),
                        self.task(generate, 1, second, 2, factor=3),
                    ],
                    fast_mode=fast_mode,
                )
                self.assertEqual(
                    [ref for _, ref in results],
                    [
                        ("first", ("data", 1), 0, 1),
                        ("first", ("data", 1), 0, 1),
                        ("first", ("data", 1), 2, 1),
                        ("first", ("data", 1), 2, 3),
                        ("second", ("data", 1), 2, 3),
                    ],
                )
                self.assertEqual(len(self.generated), 1)
                self.assertEqual(len(self.references), 4)

    def test_explicit_reference_does_not_replace_computed_cache(self):
        generate, reference = self.generator("data"), self.reference("computed")
        results = self.run_tasks(
            [
                self.task(generate, 1, reference, explicit_ref="first-explicit"),
                self.task(generate, 1, reference),
                self.task(generate, 1, reference, explicit_ref="second-explicit"),
                self.task(generate, 1, reference),
            ]
        )
        computed = ("computed", ("data", 1), 0, 1)
        self.assertEqual(
            [ref for _, ref in results],
            ["first-explicit", computed, "second-explicit", computed],
        )
        self.assertEqual(len(self.references), 1)

    def test_fast_task_without_reference_does_not_reuse_previous_reference(self):
        generate, reference = self.generator("data"), self.reference("computed")
        results = self.run_tasks(
            [
                self.task(generate, 1, reference),
                self.task(generate, 1, None),
            ],
            fast_mode=True,
        )
        self.assertIsNone(results[1][1])
        self.assertEqual(len(self.references), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
