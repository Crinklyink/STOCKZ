from __future__ import annotations

import unittest
from unittest.mock import patch

from stock_predictor.config import AppConfig
from stock_predictor.output.bot import format_bot_message, send_weekly_picks
from tests.helpers import make_candidate


class BotIntegrationTests(unittest.TestCase):
    def test_weekly_pick_message_posts_to_discord_and_telegram(self) -> None:
        candidate = make_candidate()
        config = AppConfig(
            discord_webhook_url="https://discord.example/hook",
            telegram_bot_token="token",
            telegram_chat_id="chat",
        )
        with patch("stock_predictor.output.bot.requests.post") as mocked_post:
            send_weekly_picks(config, [candidate])
        self.assertEqual(mocked_post.call_count, 2)
        self.assertIn("NVDA", format_bot_message("Weekly", [candidate]))


if __name__ == "__main__":
    unittest.main()
