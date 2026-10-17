"""Test settings. Runs before the tests are collected."""
import os

# Many requests are sent very fast during tests, so relax the rate limits here.
# (The rate limit tests change the limit themselves.)
os.environ["RATE_LIMIT_PER_MIN"] = "100000"
os.environ["LOGIN_LIMIT_PER_MIN"] = "100000"
