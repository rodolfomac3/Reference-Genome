import os

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = 2            # bump to 3–4 if CPU allows
threads = 4
timeout = 120          # allow primer3 / PDF work
graceful_timeout = 30
keepalive = 5
loglevel = "info"
