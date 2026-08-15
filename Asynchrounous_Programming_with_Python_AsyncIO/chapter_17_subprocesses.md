# Chapter 17: Subprocesses with AsyncIO

> Part 4 — I/O, Networking, and Real Systems · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 17.1 Why Async Subprocesses?

Running an external program — `git`, `ffmpeg`, a compiler, a shell script — and waiting for it to finish is just another form of waiting. The synchronous `subprocess.run()` blocks the entire thread it's called from; inside an event loop, that means blocking **every other coroutine** too, exactly per Chapter 3's cooperative-multitasking rule. `asyncio` provides async equivalents so you can run one (or many) subprocesses concurrently with everything else your program is doing.

---

## 17.2 `create_subprocess_exec` vs. `create_subprocess_shell`

- **`asyncio.create_subprocess_exec(program, *args)`** — runs a program **directly**, with arguments passed as separate strings, no shell involved. **Prefer this by default** — there's no shell parsing of your arguments, so there's no shell-injection risk even if some part of the command comes from user input.
- **`asyncio.create_subprocess_shell(cmd)`** — runs a single string through the system shell (`/bin/sh` on POSIX, `cmd.exe` on Windows), which lets you use pipes, redirects, and globbing directly in the string — but **carries real shell-injection risk** if any part of `cmd` is built from untrusted input.

```python
# Prefer this:
proc = await asyncio.create_subprocess_exec("ls", "-la", user_supplied_path)

# Over this, unless you specifically need shell features and control the input:
proc = await asyncio.create_subprocess_shell(f"ls -la {user_supplied_path}")  # risky if path is untrusted
```

---

## 17.3 Basic Usage: `communicate()`

For a "run it, wait for it to finish, get its output" pattern with bounded output size, `communicate()` is the simplest tool:

```python
import asyncio

async def main():
    proc = await asyncio.create_subprocess_exec(
        "ls", "-la",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    print(f"Exit code: {proc.returncode}")
    print(stdout.decode())

asyncio.run(main())
```

`communicate()` awaits until the process actually exits, then returns `(stdout_bytes, stderr_bytes)` — both fully buffered in memory. This is convenient, but exactly because it buffers everything, it's not the right tool for processes with very large or effectively unbounded output.

If you don't need the output at all, `await proc.wait()` simply waits for the exit code without capturing anything.

---

## 17.4 Streaming Output Incrementally

For long-running processes, or ones with large output, read from `proc.stdout` (itself a `StreamReader`, exactly like Chapter 16's network streams) incrementally instead:

```python
import asyncio

async def main():
    proc = await asyncio.create_subprocess_exec(
        "ping", "-c", "4", "example.com",
        stdout=asyncio.subprocess.PIPE,
    )
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(f"Output: {line.decode().rstrip()}")

    await proc.wait()
    print(f"Exit code: {proc.returncode}")

asyncio.run(main())
```

This prints each line the moment it arrives, rather than waiting for the whole process to finish — the same "stream, don't buffer everything" principle from Chapter 15's async generators, now applied to subprocess output.

---

## 17.5 Concurrently Piping stdin and stdout (Avoiding Deadlock)

If you need to **both** write to a subprocess's `stdin` **and** read its `stdout`/`stderr` for an interactive, long-running process, doing this **sequentially** is a classic deadlock trap: if you write a large amount of data to `stdin` before reading any `stdout`, and the subprocess's own output pipe buffer fills up while it's still trying to read more `stdin`, both sides can end up stuck waiting on each other.

**The fix: read and write concurrently**, using a `TaskGroup`, exactly like Chapter 16's send/receive pattern:

```python
import asyncio

async def write_stdin(proc, lines):
    for line in lines:
        proc.stdin.write((line + "\n").encode())
        await proc.stdin.drain()
    proc.stdin.close()

async def read_stdout(proc):
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(f"Got: {line.decode().rstrip()}")

async def main():
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c",
        "import sys\nfor line in sys.stdin:\n    sys.stdout.write('echo: ' + line)\n    sys.stdout.flush()",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(write_stdin(proc, ["hello", "world", "async is neat"]))
        tg.create_task(read_stdout(proc))

    await proc.wait()

asyncio.run(main())
```

Writing and reading happen concurrently here, so neither side's buffer can back up the other into a standstill — the exact same principle as any producer/consumer pair from Chapter 13, just with an OS process on the other end instead of an `asyncio.Queue`.

---

## 17.6 Running Multiple Subprocesses Concurrently, Bounded

Since each subprocess call is itself async, you can run many external programs concurrently — bounded by a `Semaphore` (Chapter 12) if you don't want to overwhelm CPU, disk, or memory:

```python
import asyncio

sem = asyncio.Semaphore(2)   # never more than 2 running at once

async def run_job(name, command):
    async with sem:
        print(f"{name}: starting")
        proc = await asyncio.create_subprocess_exec(*command)
        await proc.wait()
        print(f"{name}: done (exit {proc.returncode})")

async def main():
    jobs = [
        ("job-1", ["sleep", "1"]),
        ("job-2", ["sleep", "1"]),
        ("job-3", ["sleep", "1"]),
        ("job-4", ["sleep", "1"]),
    ]
    async with asyncio.TaskGroup() as tg:
        for name, command in jobs:
            tg.create_task(run_job(name, command))

asyncio.run(main())
```

This is exactly the same bounded-concurrency pattern from Chapters 12 and 17.6's fan-out chapters — now applied to real OS processes, useful for things like running a fixed number of parallel `ffmpeg` conversions or test-suite shards without saturating the machine.

---

## 17.7 Timeouts and Cancellation: The Process Doesn't Die on Its Own

Here's an important gotcha: **cancelling the `asyncio` task that's awaiting a subprocess does *not* automatically kill the underlying OS process.** The subprocess keeps running in the background, orphaned from your program's perspective, unless you explicitly terminate it yourself.

```python
import asyncio

async def main():
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", "import time; time.sleep(100)"
    )
    try:
        async with asyncio.timeout(2):
            await proc.wait()
    except TimeoutError:
        print("Timed out — killing the subprocess")
        proc.kill()          # or proc.terminate() for a gentler SIGTERM first
        await proc.wait()    # always reap it after killing, to avoid a zombie process
        print(f"Process actually exited with code {proc.returncode}")

asyncio.run(main())
```

`proc.terminate()` sends `SIGTERM` (a polite request to shut down); `proc.kill()` sends `SIGKILL` (immediate, unconditional termination). Whichever you use, **always follow it with `await proc.wait()`** — this reaps the process (collects its exit status from the OS), preventing a lingering zombie process entry.

---

## 17.8 Hands-On Exercises

**Exercise 1 — Basic `communicate()`.**
Run `ls -la` (or `dir` on Windows, adjusted accordingly) via `create_subprocess_exec`, capture output with `communicate()`, and print the return code and decoded stdout.

**Exercise 2 — Stream output incrementally.**
Run a longer command (e.g., `ping -c 5 example.com`, or a small Python script that prints and sleeps in a loop) and print each line as it arrives via `readline()`, rather than waiting for the whole thing to finish with `communicate()`.

**Exercise 3 — Concurrent stdin/stdout without deadlock.**
Implement the §17.5 example yourself. Then deliberately break it: rewrite it to write **all** stdin lines first (fully, sequentially) and only start reading stdout afterward, using a large enough number of lines (e.g., a few thousand short lines) that you can observe a stall. Explain what's happening.

**Exercise 4 — Bounded concurrent subprocess pool.**
Implement the §17.6 example with 6 jobs and `Semaphore(2)`. Print timestamps to confirm no more than 2 are ever running simultaneously.

**Exercise 5 — Timeout and clean kill.**
Implement the §17.7 example yourself. Confirm that without the `proc.kill()` + `proc.wait()` cleanup, the subprocess would keep running well past your 2-second timeout (you can verify this on POSIX systems with `ps aux | grep python3` in another terminal while your script is timing out).

---

## 17.9 Common Pitfalls

- **Using `create_subprocess_shell()` with any untrusted input in the command string.** This is a real shell-injection risk — prefer `create_subprocess_exec()` with separate argument strings whenever possible.
- **Writing all of stdin before reading any stdout (or vice versa) for a long-running interactive process.** This can deadlock once OS pipe buffers fill up — always read and write concurrently via a `TaskGroup`.
- **Assuming cancellation kills the OS process.** It doesn't — you must explicitly call `proc.terminate()`/`proc.kill()` and then `await proc.wait()` to actually stop and reap it.
- **Forgetting `await proc.wait()` after killing a process.** Without it, the process can linger as a zombie until your own process exits.
- **Using `communicate()` for processes with very large or unbounded output.** It buffers everything in memory — stream incrementally via `proc.stdout.readline()`/`.read(n)` instead for anything large or long-running.

---

## 17.10 Further Reading

- [Python docs: `asyncio` Subprocesses](https://docs.python.org/3/library/asyncio-subprocess.html)

---

**Next: Chapter 18 — Bridging Sync and Async Code** (`loop.run_in_executor`, `ThreadPoolExecutor` vs. `ProcessPoolExecutor`, and `asyncio.to_thread` — formalizing the `ainput()` trick from Chapter 16 into a general pattern for calling blocking libraries safely from async code).
