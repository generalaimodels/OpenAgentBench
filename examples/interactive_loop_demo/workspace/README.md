# Demo Workspace

This workspace is intentionally small so the interactive loop can inspect it through the local terminal tool.

Suggested commands:

```bash
ls -la
python3 -m unittest -q test_demo_stats.py
rg moving_average .
python3 -c "from demo_stats import moving_average; print(moving_average([1, 2, 3, 4], 2))"
```
