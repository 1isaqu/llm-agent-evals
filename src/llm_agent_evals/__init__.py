"""A tool-calling analytics agent with an evaluation harness.

The package has four layers that depend only downwards:

* data: `sources` (a protocol, SQL builders over a neutral contract, a fixture source);
* analytics: pure functions that turn rollup rows into the numbers the agent reports;
* agent: tool registry, run guard, prompt, model clients and the loop itself;
* evaluation: scenarios, assertion layers, metrics and the reliability ledger.
"""

__version__ = "0.1.0"
