"""Exercise the launcher's port probe against real TCP socket lifetimes."""
import errno
import importlib.util
from pathlib import Path
import socket
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "rtx4090_run_local", ROOT / "experiments/scripts/rtx4090/run_local.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherPortsTest(unittest.TestCase):
    def test_unused_port_is_released_after_probe(self):
        with socket.socket() as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = candidate.getsockname()[1]
        launcher.check_tcp_port(port, reuse_address=True)
        with socket.socket() as listener:
            listener.bind(("0.0.0.0", port))
            listener.listen(1)

    def test_active_listener_is_rejected_even_when_reusable(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "Identify the owner/state") as caught:
                launcher.check_tcp_port(port, reuse_address=True)
            self.assertEqual(caught.exception.__cause__.errno, errno.EADDRINUSE)

    def test_recently_closed_connection_allows_zmq_style_reuse(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(2)
            client.settimeout(2)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            client.connect(("127.0.0.1", port))
            connection, _ = listener.accept()
            listener.close()
            with connection:
                # Server actively closes, leaving the server port in TIME_WAIT.
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
                client.shutdown(socket.SHUT_WR)
                self.assertEqual(connection.recv(1), b"")
        with self.assertRaises(RuntimeError) as caught:
            launcher.check_tcp_port(port)
        self.assertEqual(caught.exception.__cause__.errno, errno.EADDRINUSE)
        launcher.check_tcp_port(port, reuse_address=True)


if __name__ == "__main__":
    unittest.main()
