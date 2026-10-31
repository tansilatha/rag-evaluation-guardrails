import unittest
from unittest.mock import patch

from src.guardrails import llama_guard


class LlamaGuardTests(unittest.TestCase):
    def test_safe_response_is_allowed(self):
        with patch.object(llama_guard, "_call_ollama", return_value="safe"):
            self.assertEqual(
                llama_guard.check_safety("question"),
                (True, "The input is safe."),
            )

    def test_unsafe_response_is_blocked(self):
        with patch.object(
            llama_guard, "_call_ollama", return_value="unsafe S13"
        ):
            is_safe, reason = llama_guard.check_safety(
                "answer", guard_type="output"
            )

        self.assertFalse(is_safe)
        self.assertIn("output was blocked by Llama Guard 2 as unsafe", reason)
        self.assertIn("unsafe S13", reason)

    def test_unexpected_and_empty_responses_are_blocked(self):
        for reply in ("", "unknown"):
            with self.subTest(reply=reply), patch.object(
                llama_guard, "_call_ollama", return_value=reply
            ):
                is_safe, reason = llama_guard.check_safety("question")
                self.assertFalse(is_safe)
                self.assertIn("blocked for safety", reason)

    def test_request_failure_is_blocked(self):
        with patch.object(
            llama_guard, "_call_ollama", side_effect=ConnectionError("offline")
        ):
            is_safe, reason = llama_guard.check_safety("question")

        self.assertFalse(is_safe)
        self.assertIn("request failed", reason)

    def test_output_check_uses_assistant_response_instruction(self):
        with patch.object(
            llama_guard, "_call_ollama", return_value="safe"
        ) as call_ollama:
            is_safe, reason = llama_guard.check_safety(
                "generated response", guard_type="output"
            )

        self.assertTrue(is_safe)
        self.assertEqual(reason, "The output is safe.")
        prompt = call_ollama.call_args.args[0]
        self.assertIn("assistant response", prompt.casefold())
        self.assertIn("Assistant response:\ngenerated response", prompt)

    def test_unknown_guard_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "guard_type must be"):
            llama_guard.check_safety("text", guard_type="unknown")


if __name__ == "__main__":
    unittest.main()
