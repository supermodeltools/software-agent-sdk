# View

The `View` class is responsible for representing and manipulating the subset of events that will be provided to the agent's LLM on every step.

It is closely tied to the context condensation system, and works to ensure the resulting sequence of messages are well-formed and respect the structure expected by common LLM APIs.