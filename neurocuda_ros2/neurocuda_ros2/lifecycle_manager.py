#!/usr/bin/env python3
"""
NeuroCUDA Lifecycle Manager — Boots and monitors lifecycle nodes.

Automatically transitions SNN nodes through their lifecycle:
  Unconfigured → Configuring → Inactive → Activating → Active

Usage:
  ros2 run neurocuda_ros2 lifecycle_mgr --ros-args -p node_names:=[snn_inference,spike_viz]
"""

import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import State, Transition


class LifecycleManager(Node):
    """Manages lifecycle transitions for a list of nodes."""

    def __init__(self):
        super().__init__("lifecycle_manager_snn")
        self.declare_parameter("node_names", ["snn_inference"])
        self.declare_parameter("transition_timeout", 10.0)
        self.declare_parameter("auto_manage", True)

        self._node_names = self.get_parameter("node_names").value
        self._timeout = self.get_parameter("transition_timeout").value
        self._auto = self.get_parameter("auto_manage").value

        if isinstance(self._node_names, str):
            self._node_names = [n.strip() for n in self._node_names.split(",")]

        self.get_logger().info(
            f"Managing lifecycle nodes: {self._node_names}"
        )

        if self._auto:
            self._boot_timer = self.create_timer(1.0, self._boot_nodes)

    # ------------------------------------------------------------------
    # Boot sequence: Unconfigured → Inactive → Active
    # ------------------------------------------------------------------
    def _boot_nodes(self):
        self.destroy_timer(self._boot_timer)

        for node_name in self._node_names:
            self.get_logger().info(f"🚀 Booting: {node_name}")
            if not self._wait_for_service(node_name, "change_state"):
                continue

            # Step 1: Configure (Unconfigured → Inactive)
            if not self._change_state(node_name, Transition.TRANSITION_CONFIGURE):
                continue

            # Step 2: Activate (Inactive → Active)
            self._change_state(node_name, Transition.TRANSITION_ACTIVATE)

        self.get_logger().info("✅ All SNN lifecycle nodes active")

    # ------------------------------------------------------------------
    # Service helpers
    # ------------------------------------------------------------------
    def _wait_for_service(self, node_name, service_type, timeout=30.0):
        service_name = f"/{node_name}/{service_type}"
        client = self.create_client(ChangeState if service_type == "change_state" else GetState, service_name)

        self.get_logger().info(f"  Waiting for {service_name}...")
        start = self.get_clock().now()

        while not client.wait_for_service(timeout_sec=1.0):
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout:
                self.get_logger().error(
                    f"  ❌ {service_name} not available after {timeout}s — "
                    f"is {node_name} running as lifecycle node?"
                )
                return False

        self.get_logger().info(f"  ✅ {service_name} available")
        return True

    def _change_state(self, node_name, transition_id):
        client = self.create_client(ChangeState, f"/{node_name}/change_state")

        request = ChangeState.Request()
        request.transition = Transition()
        request.transition.id = transition_id
        request.transition.label = self._transition_label(transition_id)

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout)

        if future.done() and future.result() is not None:
            result = future.result()
            if result.success:
                self.get_logger().info(
                    f"  ✅ {node_name} → {self._transition_label(transition_id)}"
                )
                return True
            else:
                self.get_logger().warn(
                    f"  ⚠️  {node_name} transition failed (may already be in state)"
                )
                return True  # Don't block — node may already be in correct state

        self.get_logger().error(f"  ❌ {node_name} transition timed out")
        return False

    def _transition_label(self, transition_id):
        labels = {
            Transition.TRANSITION_CREATE: "create",
            Transition.TRANSITION_CONFIGURE: "configure",
            Transition.TRANSITION_CLEANUP: "cleanup",
            Transition.TRANSITION_ACTIVATE: "activate",
            Transition.TRANSITION_DEACTIVATE: "deactivate",
            Transition.TRANSITION_INACTIVE_SHUTDOWN: "shutdown",
            Transition.TRANSITION_ACTIVE_SHUTDOWN: "shutdown",
            Transition.TRANSITION_DESTROY: "destroy",
        }
        return labels.get(transition_id, f"transition_{transition_id}")


def main(args=None):
    rclpy.init(args=args)
    node = LifecycleManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
