# Network Diagnosis Agent

An interactive terminal agent built with LangChain and `ChatTongyi`, using the DashScope API. Describe a network issue, then ask follow-up questions in the same conversation.

The diagnostic tools run real, read-only checks on the machine running Python: ICMP ping, system DNS resolution, interface inspection, and searches of explicitly configured log files. Tool results are included in the conversation sent to DashScope.

## Setup

Run these commands from the repository root. A separate virtual environment keeps this project's dependencies isolated from other course examples. The project has been tested with Python 3.12.

```bash
cd "example_project/network_diagnosis_agent"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements file specifies:

| Package | Version |
| --- | --- |
| `dashscope` | `1.22.1` |
| `langchain` | `1.0.2` |
| `langchain_community` | `0.4` |
| `psutil` | `>=5.9,<8` |

## Configuration

Set your DashScope API key in the terminal before starting the script:

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

| Environment variable | Behavior |
| --- | --- |
| `DASHSCOPE_API_KEY` | Preferred API key. |
| `DASHSCOPE_MODEL` | Optional model override; the current script defaults to `qwen3.8-max`. |

The default is a value configured in the source, not a guarantee that the model is available to your account. To select another model, set its DashScope model ID before running:

```bash
export DASHSCOPE_MODEL="your-enabled-model-id"
```

Choose a model available to your account that supports tool calling through this integration. The script reads environment variables directly; it does not automatically load a `.env` file.

## Run the terminal chat

From the project directory with the virtual environment activated:

```bash
python network_diagnosis_agent.py
```

Alternatively, use the existing repository environment from the repository root:

```bash
zhihullm/bin/python "handson project/network_diagnosis_agent/network_diagnosis_agent.py"
```

The terminal displays a `You:` prompt. Try:

```text
I cannot access www.example.com. The browser reports a connection timeout.
Can you check the DNS resolution too?
Check the status of interface eth1.
Search the logs for connection refused errors.
```

The agent chooses tools based on your request and prints its final response after processing. Responses are not streamed to the terminal.

| Input | Action |
| --- | --- |
| A message | Send a request with the current conversation history. |
| `/reset` | Clear conversation history and start a new discussion. |
| `exit` or `quit` | End the session. |
| Ctrl+C or end-of-input (Ctrl+D on macOS/Linux) | Exit cleanly. |
| Empty line | Wait for another input. |

History includes user messages, agent messages, tool calls, and tool results. It lives only in memory and is lost when the process exits. After a failed request, the previous successful conversation is retained; the failed turn is not saved. Long conversations are not automatically trimmed, so use `/reset` to clear old context.

## Diagnostic tools

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `ping_tool` | `target` | Runs three ICMP requests using the system ping executable, with a 10-second process timeout. Reports command output and exit code. A failed ping can mean ICMP is blocked. |
| `dns_tool` | `hostname` | Uses the system resolver in a child process with a 10-second timeout and returns IPv4/IPv6 addresses. Local host-file results may also be returned. |
| `interface_check_tool` | Optional `interface_name` | Reads link status, MTU, addresses and netmasks with psutil. Omitting the name lists all interfaces; it does not select a default route. |
| `log_analysis_tool` | `keywords`, optional `time_range` | Searches configured files for a case-insensitive literal phrase. Reads the last 1 MiB per file and returns at most 50 matches, each limited to 1,000 characters. |

Ping requires the operating system's `ping` executable (`ping6` for IPv6 on macOS). These checks do not change network configuration. An interface being up does not prove internet connectivity.

### Configure log files

Set this variable before launching the agent (example paths must be replaced with your own):

```bash
export NETWORK_DIAG_LOG_FILES="/absolute/path/application.log:/absolute/path/network.log"
```

Paths are separated by `os.pathsep`: `:` on macOS/Linux or `;` on Windows. Only configured regular files are read; without configuration, the tool returns setup instructions. The agent cannot select arbitrary file paths through tool arguments.

The default time range is `past hour`. Other accepted values are `past N minutes`, `past N hours`, `past N days`, and `all`. Filtered searches require ISO timestamps at the start of each line, optionally preceded by `[`, for example:

```text
2026-10-27T14:30:00+00:00 connection timeout
```

Timestamps without timezone information use the machine's local timezone. Matching lines without parseable timestamps are counted as skipped; `all` includes them. This tool reads text files, not macOS unified logging or the Windows event log directly. Export those logs to a text file if needed.

Known API-key values and common password/token fields are redacted before results are returned. Redaction is best-effort; configure only logs appropriate to share with the hosted model. Truncation and read errors are reported, so zero matches does not prove that no errors occurred outside the searched data.

## Code structure

- `DashScopeClientAdapter`: converts SDK responses into ordinary dictionaries, including streamed chunks. This prevents the SDK's missing-attribute behavior from masking API errors with `KeyError: 'request'` in LangChain's HTTP error handling.
- `create_network_diagnosis_chain()`: reads configuration, creates `ChatTongyi`, wraps its client, registers the four tools, and returns the LangChain agent.
- `diagnose_network_issue(issue_description)`: runs a standalone diagnosis with a fresh agent. It returns the final response, or an error string after printing a traceback.
- `run_terminal_chat()`: creates one agent and manages the interactive loop and conversation history.
- The `__main__` block starts `run_terminal_chat()` when the file is executed directly.

For a standalone call from Python in this directory:

```python
from network_diagnosis_agent import diagnose_network_issue

result = diagnose_network_issue("Check DNS for unknown.domain.xyz.")
print(result)
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Set DASHSCOPE_API_KEY` | Export a key in the same terminal used to start Python. |
| HTTP 429 / `Throttling.RateQuota` | The API is rate-limiting requests. Wait before retrying and check your account's limits. The response adapter exposes this error; it does not remove rate limits. |
| Authentication, model access, or model-not-found error | Check the API key and the model configured by `DASHSCOPE_MODEL`. |
| Connection or DNS resolution error | Check access to the DashScope service from your environment. |
| Import errors or dependency conflicts | Use the project's isolated virtual environment and install its requirements there. Older course dependencies in a shared environment can conflict with these versions. |

The terminal loop reports request errors and allows another input. A startup error, such as a missing API key, ends the program.

## Tests

From the project directory, run:

```bash
python -m unittest discover -s . -p 'test_*.py'
```

The tests cover API error handling, terminal conversation behavior, ping validation and timeouts, local DNS resolution, actual interface inspection, and log filtering/redaction using temporary files. They make no hosted model API requests. Model availability and external-host connectivity are not verified by this suite.
