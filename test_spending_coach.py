import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spending_coach import create_spending_router


class FakeClient:
    def __init__(self):
        self.response = {"summary": "Review your recorded spending.", "recommendations": [
            {"title": "Review groceries", "action": "Check your supplies before shopping.", "category": "Groceries"},
            {"title": "Make a list", "action": "Plan your next shopping trip.", "category": None},
        ]}
        self.responses = self
        self.calls = []
    def with_options(self, **kwargs): return self
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(status="completed", output_text=json.dumps(self.response))


@pytest.fixture
def setup():
    fake = FakeClient()
    app = FastAPI()
    app.include_router(create_spending_router(fake, "test-model"))
    return TestClient(app), fake


def payload():
    return {"currency": "UAH", "period_days": 30, "purchase_count": 2, "total": 100,
            "previous_total": 0, "previous_purchase_count": 0,
            "categories": [{"name": "Groceries", "amount": 100}]}


def test_grounded_server_evidence_and_no_storage(setup):
    client, fake = setup
    result = client.post("/spending-recommendations", json=payload())
    assert result.status_code == 200
    data = result.json()
    assert data["recommendations"][0]["evidence"] == "Groceries: UAH 100.00 recorded."
    assert data["currency"] == "UAH" and data["period_days"] == 30
    assert fake.calls[0]["store"] is False
    sent = json.loads(fake.calls[0]["input"])
    assert set(sent) == set(payload())


@pytest.mark.parametrize("change", [
    {"total": -1}, {"total": 101}, {"currency": "UAH USD"}, {"period_days": 10},
    {"purchase_count": 0}, {"categories": []}, {"previous_total": 10},
    {"raw_text": "must not be sent"},
    {"categories": [{"name": "Groceries", "amount": 50}, {"name": "Groceries", "amount": 50}]},
])
def test_reject_invalid_inputs_before_ai(setup, change):
    client, fake = setup
    request = payload(); request.update(change)
    assert client.post("/spending-recommendations", json=request).status_code == 422
    assert not fake.calls


def test_unknown_output_category_is_not_presented(setup):
    client, fake = setup
    fake.response["recommendations"][0]["category"] = "Invented"
    response = client.post("/spending-recommendations", json=payload())
    assert response.status_code == 502
    assert "Invented" not in response.text


def test_refusal_or_empty_output_gets_safe_error(setup):
    client, fake = setup
    fake.response = {}
    assert client.post("/spending-recommendations", json=payload()).status_code == 502
