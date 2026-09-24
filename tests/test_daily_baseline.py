"""设备每日同窗基线偏移查询测试。"""

import pytest

DAY = 86400
BASE = "2026-09-20T00:00:00Z"  # 目标窗口
FMT = "%Y-%m-%dT%H:%M:%SZ"


def ts(days_before, seconds_in_day=0):
    """BASE 前 days_before 天、当天第 seconds_in_day 秒的 ISO 时间。"""
    from datetime import datetime, timedelta, timezone

    t = datetime(2026, 9, 20, tzinfo=timezone.utc) - timedelta(days=days_before, seconds=-seconds_in_day)
    return t.strftime(FMT)


def ingest(client, device_id, events):
    return client.post("/ingest", json={"device_id": device_id, "events": events})


def configure(client, device_id, window_size_seconds=60, allowed_lateness_seconds=0):
    return client.put(
        f"/devices/{device_id}/config",
        json={
            "window_size_seconds": window_size_seconds,
            "allowed_lateness_seconds": allowed_lateness_seconds,
        },
    )


def baseline(client, device_id="dev-1", window_start=BASE, days=7, threshold=3.5):
    return client.get(
        f"/devices/{device_id}/daily-baseline",
        params={"window_start": window_start, "days": days, "threshold": threshold},
    )


def seed_history(client, device_id="dev-1", offsets=(10.0, 11.0, 12.0), seconds=10):
    """在目标窗口前若干天的 UTC 同窗位各写一个事件，并全部封存。

    offsets[i] 对应前 (i+1) 天；另用一条晚于所有历史窗的事件把历史窗封存。
    """
    events = []
    for i, value in enumerate(offsets, start=1):
        events.append({
            "event_id": f"hist-{i}",
            "event_time": ts(i, seconds),
            "value": value,
        })
    ingest(client, device_id, events)
    # 水位线推进到目标日前一天 00:05（lateness=0），封存全部历史窗
    ingest(client, device_id, [{
        "event_id": "hist-seal",
        "event_time": ts(1, 300),
        "value": 0.0,
    }])


def test_basic_offset_score_and_outlier(client):
    configure(client, "dev-1")
    # 历史同窗均值 10,11,12 => 中位数 11，MAD 1
    seed_history(client, offsets=(10.0, 11.0, 12.0))
    # 目标窗口均值 15（两个事件 14/16），封存目标窗
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 5), "value": 14.0},
        {"event_id": "t2", "event_time": ts(0, 20), "value": 16.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])

    r = baseline(client, days=5, threshold=3.5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["sample_count"] == 3
    assert body["required_samples"] == 3
    assert body["target_avg"] == pytest.approx(15.0)
    assert body["baseline_median"] == pytest.approx(11.0)
    assert body["mad"] == pytest.approx(1.0)
    assert body["offset_score"] == pytest.approx(4.0)
    assert body["outlier"] is True

    # 阈值等于分数：严格大于才越界 => 不越界
    body_eq = baseline(client, days=5, threshold=4.0).json()
    assert body_eq["offset_score"] == pytest.approx(4.0)
    assert body_eq["outlier"] is False

    # 只回看 3 天恰好 3 个样本
    assert baseline(client, days=3).json()["sample_count"] == 3


def test_samples_listed_in_offset_order_and_missing_marked(client):
    configure(client, "dev-1")
    # 只在前 1、3 天有同窗样本；第 2、4、5 天缺失，绝不用临近窗口代替
    ingest(client, "dev-1", [
        {"event_id": "h1", "event_time": ts(1, 10), "value": 10.0},
        {"event_id": "h3", "event_time": ts(3, 10), "value": 12.0},
        {"event_id": "seal-hist", "event_time": ts(1, 300), "value": 0.0},
    ])
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 11.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])

    body = baseline(client, days=5).json()
    assert body["sample_count"] == 2
    assert body["status"] == "insufficient_samples"
    assert body["baseline_median"] is None
    assert body["mad"] is None
    assert body["offset_score"] is None
    assert body["outlier"] is None

    statuses = [(s["offset_days"], s["status"], s["included"]) for s in body["samples"]]
    assert statuses == [
        (1, "sealed", True),
        (2, "missing", False),
        (3, "sealed", True),
        (4, "missing", False),
        (5, "missing", False),
    ]
    sealed = [s for s in body["samples"] if s["status"] == "sealed"]
    assert sealed[0]["avg"] == pytest.approx(10.0)
    missing = body["samples"][1]
    assert missing["avg"] is None
    assert missing["window_start"] == "2026-09-18T00:00:00Z"
    assert missing["window_end"] == "2026-09-18T00:01:00Z"


def test_insufficient_samples_filled_by_late_event_then_ok(client):
    configure(client, "dev-1")
    seed_history(client, offsets=(10.0, 12.0))  # 仅 2 个有效样本
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 11.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])

    assert baseline(client).json()["status"] == "insufficient_samples"

    # 迟到事件补上前 3 天的空窗（该窗位因水位线早已越过封存条件，事件即 late）
    r = ingest(client, "dev-1", [
        {"event_id": "late-h3", "event_time": ts(3, 10), "value": 11.0},
    ])
    assert r.json()["results"][0]["status"] == "late"

    body = baseline(client).json()
    assert body["status"] == "ok"
    assert body["sample_count"] == 3
    assert body["baseline_median"] == pytest.approx(11.0)
    assert body["mad"] == pytest.approx(1.0)
    assert body["offset_score"] == pytest.approx(0.0)
    assert body["outlier"] is False


def test_non_aligned_nearby_window_is_not_substituted(client):
    configure(client, "dev-1")
    # 前一天同窗位缺失，但相邻一分钟窗口有数据——不得充当样本
    ingest(client, "dev-1", [
        {"event_id": "n1", "event_time": ts(1, 70), "value": 10.0},
        {"event_id": "n2", "event_time": ts(1, 130), "value": 12.0},
        {"event_id": "seal-hist", "event_time": ts(1, 300), "value": 0.0},
    ])
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 99.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])

    body = baseline(client, days=3).json()
    assert body["sample_count"] == 0
    assert body["status"] == "insufficient_samples"
    assert all(s["status"] == "missing" for s in body["samples"])


def test_zero_mad_same_value_scores_zero(client):
    configure(client, "dev-1")
    seed_history(client, offsets=(10.0, 10.0, 10.0))
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 10.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])

    body = baseline(client).json()
    assert body["status"] == "ok"
    assert body["mad"] == pytest.approx(0.0)
    assert body["offset_score"] == pytest.approx(0.0)
    assert body["outlier"] is False


def test_zero_mad_different_value_score_null_and_outlier(client):
    configure(client, "dev-1")
    seed_history(client, offsets=(10.0, 10.0, 10.0))
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 42.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])

    body = baseline(client, threshold=1000.0).json()
    assert body["status"] == "ok"
    assert body["mad"] == pytest.approx(0.0)
    assert body["baseline_median"] == pytest.approx(10.0)
    assert body["target_avg"] == pytest.approx(42.0)
    assert body["offset_score"] is None
    # 阈值再大也越界：异值在 MAD=0 下无法量化但判为偏离
    assert body["outlier"] is True


def test_late_correction_to_target_reflected(client):
    configure(client, "dev-1")
    seed_history(client, offsets=(10.0, 11.0, 12.0))
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 11.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])
    body1 = baseline(client).json()
    assert body1["target_avg"] == pytest.approx(11.0)
    assert body1["outlier"] is False

    # 迟到修正拉高目标均值 => 重新查询立即反映
    ingest(client, "dev-1", [
        {"event_id": "t-late", "event_time": ts(0, 20), "value": 19.0},
    ])
    body2 = baseline(client).json()
    assert body2["target_avg"] == pytest.approx(15.0)
    assert body2["offset_score"] == pytest.approx(4.0)
    assert body2["outlier"] is True


def test_late_correction_to_history_sample_reflected(client):
    configure(client, "dev-1")
    seed_history(client, offsets=(10.0, 11.0, 12.0))
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 15.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])
    assert baseline(client).json()["offset_score"] == pytest.approx(4.0)

    # 迟到事件修正前 1 天样本（10 -> 14），基线变为中位数 12、MAD 1
    ingest(client, "dev-1", [
        {"event_id": "h-late", "event_time": ts(1, 20), "value": 18.0},
    ])
    body = baseline(client).json()
    samples = {s["offset_days"]: s for s in body["samples"]}
    assert samples[1]["avg"] == pytest.approx(14.0)
    assert body["baseline_median"] == pytest.approx(12.0)
    assert body["mad"] == pytest.approx(1.0)
    assert body["offset_score"] == pytest.approx(3.0)


def test_open_target_returns_409(client):
    configure(client, "dev-1")
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 10.0},
    ])
    r = baseline(client)
    assert r.status_code == 409


def test_missing_device_or_target_window_returns_404(client):
    configure(client, "dev-1")
    assert baseline(client, device_id="ghost").status_code == 404

    ingest(client, "dev-1", [
        {"event_id": "e1", "event_time": ts(0, 10), "value": 10.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])
    r = baseline(client, window_start="2026-09-20T00:05:00Z")
    assert r.status_code == 404


def test_window_size_not_dividing_day_returns_422(client):
    # 86400 不能被 7、13、86399 整除
    for size in (7, 13, 86399):
        configure(client, f"dev-{size}", window_size_seconds=size)
        r = baseline(client, device_id=f"dev-{size}")
        assert r.status_code == 422

    # 可整除的大小（1、60、900、86400）不因该规则报错
    configure(client, "dev-ok", window_size_seconds=900)
    assert baseline(client, device_id="dev-ok").status_code == 404


def test_param_validation(client):
    configure(client, "dev-1")
    url = "/devices/dev-1/daily-baseline"
    assert client.get(url, params={"window_start": BASE, "days": 2, "threshold": 1}).status_code == 422
    assert client.get(url, params={"window_start": BASE, "days": 31, "threshold": 1}).status_code == 422
    assert client.get(url, params={"window_start": BASE, "days": 7, "threshold": 0}).status_code == 422
    assert client.get(url, params={"window_start": BASE, "days": 7, "threshold": -1}).status_code == 422
    # 缺少必填参数
    assert client.get(url, params={"window_start": BASE, "days": 7}).status_code == 422


def test_read_only(client):
    configure(client, "dev-1")
    seed_history(client, offsets=(10.0, 11.0, 12.0))
    ingest(client, "dev-1", [
        {"event_id": "t1", "event_time": ts(0, 10), "value": 50.0},
        {"event_id": "seal", "event_time": ts(0, 65), "value": 0.0},
    ])
    before = client.get("/devices/dev-1/windows").json()
    for days in (3, 7, 30):
        baseline(client, days=days)
    baseline(client, threshold=1000.0)
    # 错误路径同样只读
    baseline(client, window_start="2026-09-20T00:05:00Z")
    after = client.get("/devices/dev-1/windows").json()
    assert after == before
