"""msolve lookup and the build driver, offline: a fake msolve and a stubbed build."""

import os
import stat
import subprocess
import sys

import pytest

from symident import install

GOOD = "[0, [65521, 2, 3, [36, 0, 65514, 1]]]:"

# the fake msolve is a shell script, which Windows cannot run
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="runs a shell script")


def fake_msolve(path, version="0.10.1", answer=GOOD):
    # a script that answers the test system of msolve_check and reports `version`
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f'#!/bin/sh\nif [ "$1" = "-V" ]; then echo "{version}"; exit 0; fi\n'
                 'while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n'
                 f"printf '%s' '{answer}' > \"$out\"\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("SYMIDENT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("SYMIDENT_MSOLVE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nobin"))
    install.msolve_check.cache_clear()
    yield
    install.msolve_check.cache_clear()


def test_cache_dir_per_platform(monkeypatch):
    assert install.cache_dir().endswith("cache")
    monkeypatch.delenv("SYMIDENT_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    monkeypatch.setattr(sys, "platform", "linux")
    assert install.cache_dir() == os.path.join("/xdg", "symident")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert install.cache_dir().endswith(os.path.join("Library", "Caches", "symident"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "/lad")
    assert install.cache_dir() == os.path.join("/lad", "symident", "cache")


@posix_only
def test_msolve_check(tmp_path):
    assert install.msolve_check(None) is None
    assert install.msolve_check(str(tmp_path / "missing")) is None
    assert install.msolve_check(fake_msolve(str(tmp_path / "a" / "msolve"))) == "0.10.1"
    assert install.msolve_check(fake_msolve(str(tmp_path / "b" / "msolve"), answer="[1]")) is None


@posix_only
def test_find_msolve_order(monkeypatch, tmp_path):
    assert install.find_msolve() == (None, None)
    # PATH: only a tested version is taken
    fake_msolve(str(tmp_path / "nobin" / "msolve"), version="0.9.0")
    assert install.find_msolve() == (None, None)
    install.msolve_check.cache_clear()
    on_path = fake_msolve(str(tmp_path / "nobin" / "msolve"))
    assert install.find_msolve() == (on_path, "PATH")
    # a complete build in the cache wins over PATH, the newest first
    cache = tmp_path / "cache"
    for v in ("0.9.9", "0.10.1"):
        fake_msolve(str(cache / f"deps-msolve-{v}" / "bin" / "msolve"))
        (cache / f"deps-msolve-{v}" / ".msolve-complete").touch()
    fake_msolve(str(cache / "deps-msolve-0.11.0" / "bin" / "msolve"))
    assert install.find_msolve() == (str(cache / "deps-msolve-0.10.1" / "bin" / "msolve"), "cache")
    env = fake_msolve(str(tmp_path / "env" / "msolve"))
    monkeypatch.setenv("SYMIDENT_MSOLVE", env)
    assert install.find_msolve() == (env, "SYMIDENT_MSOLVE")


def tools(monkeypatch, have):
    monkeypatch.setattr(install.shutil, "which", lambda t: f"/bin/{t}" if t in have else None)


@pytest.mark.parametrize("platform, have, match", [
    ("win32", {"curl", "make", "m4"}, "not supported on Windows"),
    ("linux", {"make", "m4"}, "needs curl or wget"),
    ("linux", {"wget"}, "needs make and m4"),
    ("linux", {"curl", "make"}, "needs m4 for"),
])
def test_install_msolve_preconditions(monkeypatch, platform, have, match):
    monkeypatch.setattr(sys, "platform", platform)
    tools(monkeypatch, have)
    with pytest.raises(RuntimeError, match=match):
        install.install_msolve(jobs=1)


def test_install_msolve_declined(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "platform", "linux")
    tools(monkeypatch, {"curl", "make", "m4"})
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert install.install_msolve(dir=str(tmp_path / "d"), ask=True, jobs=1) is None
    assert "nothing was downloaded" in capsys.readouterr().out
    assert not (tmp_path / "d").exists()


def stub_build(monkeypatch, returncode=0, working=True):
    # the build script replaced: it leaves a fake msolve and prints its prefix last
    real = subprocess.run

    def run(args, **kw):
        if args[0] != "sh":
            return real(args, **kw)
        prefix = os.path.join(kw["env"]["SYMIDENT_LIBS_CACHE"], "deps-msolve-" + kw["env"]["SYMIDENT_MSOLVE_VERSION"])
        fake_msolve(os.path.join(prefix, "bin", "msolve"), answer=GOOD if working else "[1]")
        open(os.path.join(prefix, ".msolve-complete"), "w").close()
        return subprocess.CompletedProcess(args, returncode, stdout=f"building\n{prefix}\n")
    monkeypatch.setattr(install.subprocess, "run", run)


@posix_only
def test_install_msolve_builds_into_the_cache(monkeypatch, tmp_path, capsys):
    tools(monkeypatch, {"curl", "make", "m4"})
    stub_build(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    path = install.install_msolve(ask=True, jobs=2)
    expected = str(tmp_path / "cache" / "deps-msolve-0.10.1" / "bin" / "msolve")
    assert path == expected
    assert "msolve installed in" in capsys.readouterr().out
    assert install.find_msolve() == (expected, "cache")


@posix_only
@pytest.mark.parametrize("returncode, working, match", [
    (1, True, "the msolve build failed"),
    (0, False, "produced no working msolve"),
])
def test_install_msolve_build_errors(monkeypatch, returncode, working, match):
    tools(monkeypatch, {"curl", "make", "m4"})
    stub_build(monkeypatch, returncode, working)
    with pytest.raises(RuntimeError, match=match):
        install.install_msolve(ask=False, quiet=True, jobs=1)
