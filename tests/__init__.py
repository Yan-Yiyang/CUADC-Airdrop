"""airdrop 离线测试套件。

全部测试都不需要真实飞控或真实图传硬件，直接运行即可：

    ./.venv/Scripts/python.exe tests/run_all.py

也可以单独跑某一个模块：

    ./.venv/Scripts/python.exe tests/test_telemetry.py
    ./.venv/Scripts/python.exe tests/test_alignment.py
    ./.venv/Scripts/python.exe tests/test_video.py

三者都是"纯 assert + 直接运行"的脚本式测试（项目当前没有 pytest，
见 AGENTS.md 的"工程缺口"），不依赖任何测试框架。
"""
