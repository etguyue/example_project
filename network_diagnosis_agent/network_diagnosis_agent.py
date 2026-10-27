import os
import re
import ipaddress
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
from typing import Optional

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi

class DashScopeClientAdapter:
    """Normalize SDK responses so LangChain can report API errors correctly."""

    def __init__(self, client):
        self.client = client

    def call(self, **kwargs):
        response = self.client.call(**kwargs)
        # SDK response objects raise KeyError for missing attributes such as
        # request, which breaks requests.HTTPError in LangChain's error handler.
        if kwargs.get("stream"):
            return (dict(chunk) for chunk in response)
        return dict(response)


def _validate_host(value: str) -> str:
    """Accept a single hostname or IP address, never command-line options."""
    value = value.strip()
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass
    value = value.encode("idna").decode("ascii")
    if len(value) > 253 or not all(
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
        for label in value.rstrip(".").split(".")
    ):
        raise ValueError("Provide a hostname or IP address, without a URL, port, or options.")
    return value


@tool
def ping_tool(target: str) -> str:
    """Send three real ICMP echo requests to a hostname or IP, with a 10-second limit."""
    try:
        target = _validate_host(target)
        windows = platform.system() == "Windows"
        command = ["ping", "-n" if windows else "-c", "3", target]
        if ":" in target and platform.system() == "Darwin":
            command[0] = "ping6"
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        output = (result.stdout + result.stderr).strip()[:6000]
        return f"Ping exit code: {result.returncode}\n{output}\nA failed ping can also mean ICMP is blocked."
    except subprocess.TimeoutExpired:
        return "Ping timed out after 10 seconds; the host may be unreachable or ICMP may be blocked."
    except (OSError, ValueError) as exc:
        return f"Unable to run ping: {exc}"


@tool
def dns_tool(hostname: str) -> str:
    """Resolve a hostname using the system resolver, returning IPv4/IPv6 addresses within 10 seconds."""
    try:
        hostname = _validate_host(hostname)
        # Use a child process so a stalled system resolver can be terminated.
        code = "import socket,sys,json; print(json.dumps(sorted({r[4][0] for r in socket.getaddrinfo(sys.argv[1], None)})))"
        result = subprocess.run(
            [sys.executable, "-c", code, hostname], capture_output=True, text=True, timeout=10
        )
        if result.returncode:
            return f"DNS resolution failed for {hostname}: {result.stderr.strip().splitlines()[-1]}"
        return f"Addresses for {hostname}: {', '.join(json.loads(result.stdout))}"
    except subprocess.TimeoutExpired:
        return "DNS resolution timed out after 10 seconds."
    except (OSError, ValueError) as exc:
        return f"DNS lookup failed: {exc}"


@tool
def interface_check_tool(interface_name: Optional[str] = None) -> str:
    """Read actual interface addresses, link status and MTU; omit the name to list all interfaces."""
    try:
        addresses = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        names = [interface_name] if interface_name else sorted(addresses)
        if interface_name and interface_name not in addresses:
            return f"Unknown interface '{interface_name}'. Available: {', '.join(sorted(addresses))}"
        output = []
        for name in names:
            status = stats.get(name)
            output.append(f"{name}: " + (
                f"{'up' if status.isup else 'down'}, MTU={status.mtu}" if status else "status unavailable"
            ))
            for address in addresses[name]:
                if address.family in (socket.AF_INET, socket.AF_INET6):
                    output.append(f"  IP={address.address}, netmask={address.netmask}")
        return "\n".join(output) or "No interfaces found."
    except (OSError, psutil.Error) as exc:
        return f"Interface inspection failed: {exc}"


def _redact_log(line: str) -> str:
    for name in ("DASHSCOPE_API_KEY", "ALIYUN_API_KEY"):
        secret = os.getenv(name)
        if secret:
            line = line.replace(secret, "[REDACTED]")
    line = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", line)
    return re.sub(
        r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]", line,
    )


@tool
def log_analysis_tool(keywords: str, time_range: Optional[str] = "past hour") -> str:
    """Search configured NETWORK_DIAG_LOG_FILES for a case-insensitive literal phrase.

    time_range accepts 'past hour', 'past N minutes/hours/days', or 'all'.
    Time filtering requires ISO timestamps at the start of each line; naive
    timestamps use local time. Only the last 1 MiB per file is read and at most
    50 matches are returned. Files are configured by the user, not the agent.
    """
    paths = os.getenv("NETWORK_DIAG_LOG_FILES", "").split(os.pathsep)
    paths = [Path(p).expanduser() for p in paths if p.strip()]
    if not paths:
        return "No log files configured. Set NETWORK_DIAG_LOG_FILES to file paths separated by os.pathsep (':' on macOS/Linux)."
    if not keywords.strip():
        return "Provide a non-empty search phrase."
    period = (time_range or "past hour").strip().lower()
    cutoff = None
    if period != "all":
        if period == "past hour":
            period = "past 1 hour"
        match = re.fullmatch(r"past ([1-9]\d*) (minute|hour|day)s?", period)
        if not match:
            return "Use 'past hour', 'past N minutes/hours/days', or 'all' for time_range."
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(**{match[2] + "s": int(match[1])})
        except OverflowError:
            return "The requested time range is too large."
    matches, notes = [], []
    skipped = 0
    for path in paths:
        try:
            if not path.is_file():
                notes.append(f"Not a regular file: {path}")
                continue
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                start = max(0, size - 1024 * 1024)
                stream.seek(start)
                if start:
                    stream.readline()
                    notes.append(f"Only the last 1 MiB was searched: {path}")
                lines = stream.read(1024 * 1024).decode("utf-8", errors="replace").splitlines()
            for line in lines:
                if keywords.casefold() not in line.casefold():
                    continue
                if cutoff is not None:
                    timestamp = re.match(r"^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:\d{2})?)", line)
                    try:
                        if not timestamp:
                            raise ValueError("missing timestamp")
                        when = datetime.fromisoformat(timestamp[1].replace(",", ".").replace("Z", "+00:00"))
                        if when.astimezone(timezone.utc) < cutoff:
                            continue
                    except ValueError:
                        skipped += 1
                        continue
                matches.append(f"{path.name}: {_redact_log(line)[:1000]}")
                if len(matches) == 50:
                    notes.append("Stopped at 50 matches; additional entries may exist.")
                    break
        except OSError as exc:
            notes.append(f"Unable to read {path}: {exc}")
        if len(matches) == 50:
            break
    if skipped:
        notes.append(f"Skipped {skipped} matching lines without a valid ISO timestamp; use 'all' to include them.")
    return "\n".join([f"Found {len(matches)} matching log entries.", *matches, *notes])


# --- Create the agent and tool chain ---

def create_network_diagnosis_chain():
    """Create the agent executor for network fault diagnosis."""
    # 1. Define the tool list (functions decorated with @tool)
    tools = [
        ping_tool,
        dns_tool,
        interface_check_tool,
        log_analysis_tool
        # Add any additional tools here (such as Traceroute or Get Config)
    ]

    # 2. Initialize the language model (use a ChatModel to support tool calling)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY before running the agent.")
    llm = ChatTongyi(
        model_name=os.getenv("DASHSCOPE_MODEL", "qwen3.8-max"),
        dashscope_api_key=api_key,
    )
    llm.client = DashScopeClientAdapter(llm.client)

    # 3. Create the agent (LangChain 1.x syntax)
    agent = create_agent(llm, tools)

    return agent

# --- Example: use the agent for network diagnosis tasks ---

def diagnose_network_issue(issue_description: str):
    """
    Use the network diagnosis agent to investigate a user-reported issue.

    Args:
        issue_description: Network issue described by the user.
    Returns:
        The agent's final diagnosis.
    """
    try:
        print(f"\n--- Starting diagnosis task ---")
        print(f"User issue: {issue_description}")
        agent = create_network_diagnosis_chain()
        # LangChain 1.x syntax: use the messages format
        result = agent.invoke({"messages": [("user", issue_description)]})
        # result is a dictionary containing a messages list; the last message is the agent response
        return result["messages"][-1].content
    except Exception as e:
        # Print detailed error information to help with debugging
        import traceback
        print(f"An error occurred while processing the diagnosis task: {str(e)}")
        traceback.print_exc()
        return f"Error processing the diagnosis task: {str(e)}"

def run_terminal_chat():
    """Chat with the agent, keeping conversation history until reset or exit."""
    print("Network diagnosis agent (live diagnostic tools).")
    print("Describe your issue. Commands: /reset, exit, quit. Ctrl+C also exits.")
    try:
        agent = create_network_diagnosis_chain()
    except Exception as exc:
        print(f"Unable to start the agent: {exc}")
        return

    messages = []
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            if user_input.lower() == "/reset":
                messages = []
                print("Conversation cleared.")
                continue

            print("Agent is thinking...")
            try:
                result = agent.invoke({
                    "messages": [*messages, ("user", user_input)]
                })
                # Retain tool calls and their results as well as chat messages.
                messages = result["messages"]
                print(f"\nAgent: {messages[-1].content}")
            except Exception as exc:
                print(f"Request failed: {exc}")
                print("Please try again. Your previous conversation is preserved.")
    except (EOFError, KeyboardInterrupt):
        print()
    print("Goodbye!")


# --- Main entry point ---
if __name__ == "__main__":
    run_terminal_chat()
