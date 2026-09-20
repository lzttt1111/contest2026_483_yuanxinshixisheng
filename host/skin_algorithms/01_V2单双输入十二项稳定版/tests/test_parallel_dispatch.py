from __future__ import annotations

import itertools
import unittest

from src import worker


ALGORITHMS = ("redness", "spots", "brown", "texture", "pores")


class _FakeAsyncResult:
    def __init__(self, task_id: str):
        self.id = task_id


class _FakeCelerySender:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._sequence = itertools.count(1)

    def send_task(self, name, args, queue):
        self.calls.append({"name": name, "args": args, "queue": queue})
        return _FakeAsyncResult(f"celery-{next(self._sequence)}")


class ParallelDispatchContractTest(unittest.TestCase):
    def _task_ids(self, prefix: str = "record") -> dict[str, str]:
        return {
            algorithm: f"{prefix}-{algorithm}"
            for algorithm in ALGORITHMS
        }

    def test_dispatches_five_independent_single_algorithm_tasks(self) -> None:
        sender = _FakeCelerySender()
        submitted = worker.submit_independent_tasks(
            self._task_ids(),
            "uploads/shared-face.jpg",
            "dermavision/v1/",
            app=sender,
        )

        self.assertEqual(tuple(submitted), ALGORITHMS)
        self.assertEqual(len(sender.calls), 5)
        self.assertEqual(
            {item["args"][0] for item in sender.calls},
            set(self._task_ids().values()),
        )
        self.assertEqual(
            {item["args"][1] for item in sender.calls},
            {"uploads/shared-face.jpg"},
        )
        for algorithm, item in zip(ALGORITHMS, sender.calls):
            self.assertEqual(item["name"], "dermavision.analyze_image")
            # 五项检测各自进入独立 direct queue，生产环境可由五个
            # concurrency=1 的 Worker 真正并行消费，不能退回公共串行队列。
            self.assertEqual(item["queue"], algorithm)
            self.assertEqual(item["args"][3], [algorithm])
            self.assertEqual(
                submitted[algorithm]["record_id"],
                self._task_ids()[algorithm],
            )
            self.assertTrue(
                submitted[algorithm]["celery_task_id"].startswith("celery-")
            )

    def test_requires_exactly_five_unique_record_ids(self) -> None:
        sender = _FakeCelerySender()
        incomplete = self._task_ids()
        incomplete.pop("pores")
        with self.assertRaisesRegex(ValueError, "算法集合不完整"):
            worker.submit_independent_tasks(
                incomplete,
                "uploads/shared-face.jpg",
                app=sender,
            )

        duplicate = self._task_ids()
        duplicate["pores"] = duplicate["texture"]
        with self.assertRaisesRegex(ValueError, "不同的 task_id"):
            worker.submit_independent_tasks(
                duplicate,
                "uploads/shared-face.jpg",
                app=sender,
            )
        self.assertEqual(sender.calls, [])

    def test_rejects_task_ids_that_collide_after_path_normalization(self) -> None:
        sender = _FakeCelerySender()
        task_ids = self._task_ids()
        task_ids["redness"] = "record:redness"
        task_ids["spots"] = "record?redness"
        with self.assertRaisesRegex(ValueError, "路径冲突"):
            worker.submit_independent_tasks(
                task_ids,
                "uploads/shared-face.jpg",
                app=sender,
            )
        self.assertEqual(sender.calls, [])

    def test_twenty_groups_keep_all_one_hundred_records_isolated(self) -> None:
        sender = _FakeCelerySender()
        all_record_ids: set[str] = set()
        for group_index in range(20):
            submitted = worker.submit_independent_tasks(
                self._task_ids(prefix=f"group-{group_index:02d}"),
                "uploads/shared-face.jpg",
                app=sender,
            )
            all_record_ids.update(
                item["record_id"] for item in submitted.values()
            )

        self.assertEqual(len(sender.calls), 100)
        self.assertEqual(len(all_record_ids), 100)
        self.assertTrue(
            all(len(item["args"][3]) == 1 for item in sender.calls)
        )

    def test_worker_concurrency_defaults_to_at_least_five(self) -> None:
        self.assertGreaterEqual(worker.celery_app.conf.worker_concurrency, 5)


if __name__ == "__main__":
    unittest.main()
