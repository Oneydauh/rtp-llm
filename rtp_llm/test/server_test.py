from unittest import TestCase, main

from rtp_llm.start_server import main as server_main


class ServerTest(TestCase):
    def test_simple(self):
        server_main()


if __name__ == "__main__":
    main()
