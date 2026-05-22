from pathlib import Path


def test_demo_runbook_covers_story_5_3_required_paths():
    runbook = Path("docs/demo-runbook.md").read_text(encoding="utf-8")

    required_snippets = [
        "portfolio-grade local demo",
        "not a production CRM, refund, billing, or ticketing integration",
        "uv sync --frozen",
        "USE_FAKE_MODEL=true",
        "DEFAULT_MODEL=fake",
        "Leaving live credentials unset is not enough by itself",
        "uv run python src/run_service.py",
        "uv run python -m streamlit run src/streamlit_app.py --server.port 8501 --server.address 127.0.0.1",
        "AGENT_URL=http://127.0.0.1:8080",
        "NO_PROXY=127.0.0.1,localhost,0.0.0.0",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8080/health",
        "http://127.0.0.1:8080/info",
        "http://127.0.0.1:8080/docs",
        "我忘记密码了，怎么重置？",
        "我的订单有问题",
        "订单号 #A100 支付成功但服务无法使用，页面报错 E401，有截图",
        "你们处理太慢了，我要投诉",
        "客户被重复扣费，要求马上退款",
        "approve",
        "reject",
        "fallback_rate",
        "models_used",
        "live_model_verification.verified",
        "qualitative_examples",
        "mismatches",
        "fallback-only",
        "fallback_rate: 0.0",
        "要求修改",
        "Free-form approval text that is not an explicit approve/reject is treated as a modification request",
    ]

    missing = [snippet for snippet in required_snippets if snippet not in runbook]
    assert not missing, f"Missing required runbook snippets: {missing}"


def test_docs_index_links_to_demo_runbook():
    index = Path("docs/index.md").read_text(encoding="utf-8")

    assert "[demo-runbook.md](./demo-runbook.md)" in index
