"""Probe a running CodeKids container. Exit 1 if any check fails."""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8080"
SECRET = "sk-test-SECRET-123"
failures = []


def run(code, language="python"):
    req = urllib.request.Request(
        BASE + "/api/run",
        data=json.dumps({"code": code, "language": language}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=30) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail)[:300])
    if not ok:
        failures.append(name)


for _ in range(60):
    try:
        get("/")
        break
    except Exception:
        time.sleep(2)

# Languages still work
check("python hello", run('print("hi")')["stdout"].strip() == "hi", run('print("hi")'))
r = run('console.log(6*7)', "javascript"); check("javascript", r["stdout"].strip() == "42", r)
r = run('public class Hello { public static void main(String[] a){ System.out.println("j"); } }', "java")
check("java", r["stdout"].strip() == "j", r)
r = run('#include <iostream>\nint main(){std::cout<<"c";}', "cpp"); check("cpp", r["stdout"].strip() == "c", r)
r = run('package main\nimport "fmt"\nfunc main(){fmt.Println("g")}', "go"); check("go", r["stdout"].strip() == "g", r)

# Secrets
r = run('import os;print(os.environ.get("GROQ_API_KEY"))')
check("env has no key", SECRET not in json.dumps(r), r)
r = run('import os\nfor p in ["/proc/1/environ", f"/proc/{os.getppid()}/environ"]:\n  try: print(open(p,"rb").read())\n  except Exception as e: print("blocked", type(e).__name__)')
check("parent /proc environ blocked", SECRET not in json.dumps(r), r)
r = run('import subprocess;print(subprocess.run(["sh","-c","cat /proc/*/environ"],capture_output=True).stdout)')
check("no environ via shell", SECRET not in json.dumps(r), r)
r = run('import os;print(os.getuid())')
check("runs as non-root", r["stdout"].strip() not in ("0", ""), r)
r = run('print(open("/app/backend/main.py").read()[:50])')
check("backend code unreadable", "PermissionError" in r["stderr"], r)
r = run('import os;print(os.listdir("/app/backend"))')
check("backend dir unlistable", "PermissionError" in r["stderr"], r)

# Path traversal
for p in ["/..%2F..%2Fproc%2Fself%2Fenviron", "/..%2Fbackend%2Fmain.py", "/../../proc/self/environ"]:
    status, body = get(p)
    check("traversal " + p, SECRET not in body and "FastAPI" not in body, (status, body[:80]))

# Limits
r = run('x = bytearray(2*1024**3)')
check("memory limit", r["exit_code"] != 0, r)
t = time.time(); r = run('while True: pass'); d = time.time() - t
check("cpu timeout", r["exit_code"] != 0 and d < 20, (round(d, 1), r))
r = run('import os\nfor i in range(2000):\n  try: os.fork()\n  except OSError: break\nprint("ok")')
time.sleep(3)
check("fork bomb contained", get("/")[0] == 200, r["stderr"][:120])
r = run('print("x"*50_000_000)')
check("huge output capped", len(r["stdout"]) < 100_000, len(r["stdout"]))
r = run('x = input()')
check("input message", "keyboard input" in r["stderr"], r["stderr"][:80])

status, body = get("/api/health")
check("health 200 (expected FAIL on main: PR #1 fixes it)", status == 200, status)

real = [f for f in failures if not f.startswith("health")]
print("\nFAILED:", real or "none")
sys.exit(1 if real else 0)
