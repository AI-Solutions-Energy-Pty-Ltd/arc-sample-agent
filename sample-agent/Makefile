.PHONY: sync run task

sync:
	uv sync

run:
	uv run python main.py

task:
	@if [ -n "$(TASK)" ]; then uv run python main.py --task $(TASK); elif [ -n "$(SPEC)" ]; then uv run python main.py --spec $(SPEC); else echo "usage: make task SPEC=notification_raise or make task TASK=<task_code>"; exit 1; fi
