# Chapter 16: Streams — TCP Clients and Servers

> Part 4 — I/O, Networking, and Real Systems · Course: Mastering Asynchronous Programming in Python (AsyncIO)

---

## 16.1 The Streams API: A Friendly Layer Over Sockets

asyncio's **streams** API (`StreamReader`/`StreamWriter`) is a high-level, `await`-friendly layer over raw sockets — you get simple `read`/`write` calls instead of manually juggling `selectors` and callbacks. (There's an even lower-level Transport/Protocol API underneath this, for advanced custom-protocol needs — out of scope for this course, but worth knowing it exists if you ever need that level of control.)

Two entry points cover almost everything:

- **`asyncio.open_connection(host, port)`** — connect as a client, returns `(reader, writer)`
- **`asyncio.start_server(client_connected_cb, host, port)`** — listen as a server; your callback is invoked with `(reader, writer)` for **each** incoming connection, automatically scheduled as its own concurrent task

---

## 16.2 A Simple TCP Client

```python
import asyncio

async def main():
    reader, writer = await asyncio.open_connection('example.com', 80)

    request = (
        b"GET / HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Connection: close\r\n\r\n"
    )
    writer.write(request)
    await writer.drain()   # respect backpressure — see §16.5

    while True:
        line = await reader.readline()
        if not line:   # empty bytes means the connection closed (EOF)
            break
        print(line.decode().rstrip())

    writer.close()
    await writer.wait_closed()

asyncio.run(main())
```

**Reading options**, depending on your protocol's shape:

| Method | Behavior |
|---|---|
| `await reader.readline()` | Reads up to and including the next `\n` |
| `await reader.read(n)` | Reads up to `n` bytes (may return fewer) |
| `await reader.readexactly(n)` | Reads **exactly** `n` bytes, or raises `IncompleteReadError` if the connection closes first |
| `await reader.readuntil(separator)` | Reads until a custom byte-string separator is found |

**`writer.write(data)` is not itself a coroutine** — it buffers the data synchronously and returns immediately. The corresponding `await writer.drain()` is what actually respects backpressure (more on this in §16.5).

---

## 16.3 A Simple TCP Server

```python
import asyncio

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"Connected: {addr}")
    try:
        while True:
            data = await reader.readline()
            if not data:
                break
            message = data.decode().rstrip()
            print(f"Received from {addr}: {message}")
            writer.write(f"echo: {message}\n".encode())
            await writer.drain()
    finally:
        print(f"Disconnected: {addr}")
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', 8888)
    addr = server.sockets[0].getsockname()
    print(f"Serving on {addr}")
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

The key thing to notice: `start_server()` schedules `handle_client()` as its **own independent task for every connection**, automatically. You never manually call `create_task()` here — the server handles any number of simultaneous clients concurrently just by virtue of each connection getting its own coroutine, cooperatively multitasked exactly as you'd expect by now.

---

## 16.4 Building a Mini Chat Server

Extending the echo server into a broadcast-style chat server just means tracking connected clients and writing to all of them (except the sender) whenever one sends a message:

```python
import asyncio

clients: set[asyncio.StreamWriter] = set()

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    clients.add(writer)
    print(f"{addr} joined ({len(clients)} total)")

    try:
        while True:
            data = await reader.readline()
            if not data:
                break
            message = data.decode().rstrip()

            # Broadcast to everyone else, concurrently rather than one at a time —
            # so one slow client's drain() doesn't delay delivery to the others.
            recipients = [c for c in clients if c is not writer]
            await asyncio.gather(
                *(_send(c, f"{addr}: {message}\n") for c in recipients),
                return_exceptions=True,
            )
    finally:
        clients.discard(writer)
        print(f"{addr} left ({len(clients)} total)")
        writer.close()
        await writer.wait_closed()

async def _send(writer, message):
    writer.write(message.encode())
    await writer.drain()

async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', 8888)
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

Note the `clients` set is mutated with plain `.add()`/`.discard()` calls — no `await` between checking and mutating it — so per Chapter 12's rule, this is safe without a `Lock`: those operations run atomically between suspension points. The broadcast itself, however, *does* involve multiple `await`s (one `drain()` per recipient), so it's deliberately done with `gather()` to send to everyone concurrently, rather than a sequential `for` loop that would let one slow client hold up delivery to everyone after them.

---

## 16.5 Backpressure Over the Network: `writer.drain()`

This is the network-level version of Chapter 13's queue backpressure lesson. `writer.write()` buffers data immediately, without waiting — if you call it repeatedly in a tight loop without ever `await`-ing `drain()`, and the receiving end is slow to read, that buffer can grow **unboundedly**, exactly like an unbounded `Queue`.

`await writer.drain()` suspends the caller if the write buffer has grown past an internal high-water mark, resuming once the OS/peer has caught up and drained enough of it. The fix is simple and easy to forget: **always `await writer.drain()` after `write()`** in any loop that writes more than a small, one-off amount of data.

---

## 16.6 A Companion Interactive Client

To actually use the chat server interactively, the client needs to **read incoming messages and send outgoing ones concurrently** — not sequentially, since either could happen at any moment. Since Python's built-in `input()` is a blocking call, it needs to run in a thread executor rather than directly in the event loop (a brief preview of Chapter 18's sync/async bridging):

```python
import asyncio

async def ainput(prompt=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)

async def receive_loop(reader):
    while True:
        line = await reader.readline()
        if not line:
            print("Server closed the connection")
            break
        print(line.decode().rstrip())

async def send_loop(writer):
    while True:
        message = await ainput()
        writer.write((message + "\n").encode())
        await writer.drain()

async def main():
    reader, writer = await asyncio.open_connection('127.0.0.1', 8888)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(receive_loop(reader))
        tg.create_task(send_loop(writer))

asyncio.run(main())
```

Both directions run concurrently under one `TaskGroup` — you can be typing a message while a new one from another client arrives and prints, without either blocking the other.

---

## 16.7 Hands-On Exercises

**Exercise 1 — Echo server and client.**
Run the §16.3 echo server, then connect to it with the §16.2-style client (adjusted to send a line and read the echoed response instead of an HTTP request). Confirm the round trip works.

**Exercise 2 — The chat server, with multiple clients.**
Run the §16.4 chat server, then connect with the §16.6 interactive client from **three separate terminal windows**. Confirm a message typed in one appears in the other two, but not echoed back to the sender.

**Exercise 3 — Observe backpressure.**
Modify the server to write a large amount of data (e.g., a 10 MB string) to a client in a tight loop **without** calling `drain()`. Use `writer.transport.get_write_buffer_size()` to print the buffer size growing. Then add `await writer.drain()` back in and confirm the buffer stays bounded instead.

**Exercise 4 — Length-prefixed framing (stretch).**
Newline-delimited messages (`readline()`) don't work for arbitrary binary data that might contain `\n` bytes. Implement a simple length-prefixed protocol instead: before sending a message, send a 4-byte big-endian integer giving its length (`struct.pack(">I", len(data))`), then the message bytes themselves. On the reading side, use `await reader.readexactly(4)` to get the length, then `await reader.readexactly(length)` to get the exact payload — no line-ending characters involved at all.

**Exercise 5 — Graceful shutdown.**
Add a `"/quit"` command to the chat client that closes the writer cleanly (`writer.close()` + `await writer.wait_closed()`) and exits the `TaskGroup` (hint: you'll need to cancel the sibling `receive_loop` task once you decide to quit, similar to Chapter 13's worker-pool shutdown pattern).

---

## 16.8 Common Pitfalls

- **Forgetting `await writer.drain()`.** Without it, a fast writer facing a slow reader can grow its buffer without bound — the exact backpressure problem from Chapter 13, now at the network layer.
- **Assuming `writer.write()` is itself async.** It isn't — it buffers synchronously and returns immediately; only `drain()` actually suspends.
- **Using `readline()` on a protocol that isn't strictly newline-delimited.** Binary data or arbitrary bytes containing `\n` will corrupt line-based parsing — use `readexactly()` with explicit length prefixes instead.
- **Broadcasting to multiple clients sequentially.** A `for` loop awaiting each recipient's `drain()` one at a time lets a single slow client delay delivery to everyone after it — use `gather()` (or a `TaskGroup`) to send concurrently instead.
- **Not calling `writer.close()` + `await writer.wait_closed()`.** Skipping this leaves connections lingering rather than cleanly shutting down.

---

## 16.9 Further Reading

- [Python docs: `asyncio` Streams](https://docs.python.org/3/library/asyncio-stream.html)

---

**Next: Chapter 17 — Subprocesses with AsyncIO** (`asyncio.create_subprocess_exec`/`_shell`, and concurrently piping stdin/stdout/stderr without deadlocking).
