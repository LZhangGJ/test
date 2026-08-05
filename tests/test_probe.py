from tamoe.execution.probe import _shared_path_status


def test_shared_path_stays_unresolved_without_all_hosts() -> None:
    hosts = {
        "a": {
            "probe_status": "SUCCEEDED",
            "tamoe_environment": {"TAMOE_PROJECT_ROOT": "/shared/project"},
            "paths": {"cwd": {"path": "/homes/user", "inode": 1}},
            "shared_mounts": {"stdout": "/homes server:/homes nfs4"},
        },
        "b": {"probe_status": "FAILED"},
    }
    status = _shared_path_status(hosts)
    assert status["status"] == "UNRESOLVED"
    assert status["confirmed_project_root"] is None
