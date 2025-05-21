import requests
import os
import pytest

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/init")


def test_initialise_user_new():
    user_id = "endpoint_test_user"
    response = requests.post(f"{BASE_URL}/user", json={"user_id": user_id})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data and data["error"] == ""
    assert "user_exists" in data and data["user_exists"] is False
    assert "config" in data
    assert "settings" in data["config"]


def test_initialise_user_existing():
    user_id = "endpoint_test_user_existing"
    # First create the user
    requests.post(f"{BASE_URL}/user", json={"user_id": user_id})
    # Now call again to check user_exists True
    response = requests.post(f"{BASE_URL}/user", json={"user_id": user_id})
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == ""
    assert data["user_exists"] is True
    assert "config" in data


def test_initialise_user_with_config():
    user_id = "endpoint_test_user_config"
    payload = {
        "user_id": user_id,
        "style": "endpoint_style",
        "agent_description": "endpoint_agent",
        "end_goal": "endpoint_goal",
        "branch_initialisation": "empty",
    }
    response = requests.post(f"{BASE_URL}/user", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == ""
    assert data["user_exists"] is False
    assert data["config"]["style"] == "endpoint_style"
    assert data["config"]["agent_description"] == "endpoint_agent"
    assert data["config"]["end_goal"] == "endpoint_goal"
    assert data["config"]["branch_initialisation"] == "empty"


def test_initialise_user_invalid_payload():
    # Missing user_id
    response = requests.post(f"{BASE_URL}/user", json={})
    assert response.status_code in (
        400,
        422,
    )  # FastAPI returns 422 for validation errors


def test_initialise_tree_success():
    user_id = "endpoint_test_tree_user"
    conversation_id = "endpoint_test_tree_convo"
    # Ensure user exists
    requests.post(f"{BASE_URL}/user", json={"user_id": user_id})
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "low_memory": True,
    }
    response = requests.post(f"{BASE_URL}/tree", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == ""
    assert data["conversation_id"] == conversation_id
    assert "tree" in data


def test_initialise_tree_user_not_exist():
    payload = {
        "user_id": "nonexistent_user",
        "conversation_id": "nonexistent_convo",
        "low_memory": True,
    }
    response = requests.post(f"{BASE_URL}/tree", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert data["error"] != ""
    assert data["conversation_id"] == "nonexistent_convo"
    assert "tree" in data


def test_initialise_tree_invalid_payload():
    # Missing required fields
    response = requests.post(f"{BASE_URL}/tree", json={})
    assert response.status_code in (400, 422)
