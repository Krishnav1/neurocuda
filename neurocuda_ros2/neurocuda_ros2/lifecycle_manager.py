#!/usr/bin/env python3
"""
NeuroCUDA Lifecycle Manager — Boots and monitors lifecycle nodes.

Automatically transitions SNN nodes through their lifecycle:
  Unconfigured → Inactive → Active

Key behavior:
  - Polls get_state after each transition to CONFIRM the state change
  - Retries transitions with backoff (handles slow model-loading in on_configure)
  - Reports honest PASS/FAIL per node per transition

Usage:
  ros2 run neurocuda_ros2 lifecycle_mgr --ros-args -p node_names:=[snn_inference,spike_viz]
"""

import time
import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import State, Transition


# State ID constants (from lifecycle_msgs/msg/State)
STATE_UNKNOWN      = 0   # State.PRIMARY_STATE_UNKNOWN
STATE_UNCONFIGURED = 1   # State.PRIMARY_STATE_UNCONFIGURED
STATE_INACTIVE     = 2   # State.PRIMARY_STATE_INACTIVE
STATE_ACTIVE       = 3   # State.PRIMARY_STATE_ACTIVE
STATE_FINALIZED    = 4   # State.PRIMARY_STATE_FINALIZED


class LifecycleManager(Node):
    """Manages lifecycle transitions for a list of nodes."""

    def __init__(self):
        super().__init__("lifecycle_manager_snn")
        self.declare_parameter("node_names", ["snn_inference"])
        self.declare_parameter("transition_timeout", 30.0)
        self.declare_parameter("state_poll_interval", 0.5)
        self.declare_parameter("max_state_wait", 60.0)
        self.declare_parameter("auto_manage", True)

        self._node_names = self.get_parameter("node_names").value
        self._timeout = self.get_parameter("transition_timeout").value
        self._poll_interval = self.get_parameter("state_poll_interval").value
        self._max_state_wait = self.get_parameter("max_state_wait").value
        self._auto = self.get_parameter("auto_manage").value

        if isinstance(self._node_names, str):
            self._node_names = [n.strip() for n in self._node_names.split(",")]

        self.get_logger().info(
            f"Managing lifecycle nodes: {self._node_names}"
        )
        self.get_logger().info(
            f"  timeout={self._timeout}s poll_interval={self._poll_interval}s max_state_wait={self._max_state_wait}s"
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
                self.get_logger().error(f"  ❌ {node_name}: change_state service never appeared — SKIPPING")
                continue
            if not self._wait_for_service(node_name, "get_state"):
                self.get_logger().error(f"  ❌ {node_name}: get_state service never appeared — SKIPPING")
                continue

            # Check current state first
            current_state = self._get_current_state(node_name)
            self.get_logger().info(f"  📍 Current state: {self._state_label(current_state)} (id={current_state})")

            # Step 1: Configure (Unconfigured → Inactive)
            if current_state == STATE_UNCONFIGURED:
                if not self._change_state_and_verify(
                    node_name,
                    Transition.TRANSITION_CONFIGURE,
                    expected_state=STATE_INACTIVE
                ):
                    self.get_logger().error(
                        f"  ❌ {node_name}: configure FAILED — node stuck in "
                        f"{self._state_label(self._get_current_state(node_name))} — SKIPPING"
                    )
                    continue
            elif current_state == STATE_INACTIVE:
                self.get_logger().info(f"  ✅ {node_name} already INACTIVE — skipping configure")
            else:
                self.get_logger().warn(
                    f"  ⚠️  {node_name} in unexpected state "
                    f"{self._state_label(current_state)} — attempting configure anyway"
                )
                if not self._change_state_and_verify(
                    node_name, Transition.TRANSITION_CONFIGURE,
                    expected_state=STATE_INACTIVE
                ):
                    self.get_logger().error(f"  ❌ {node_name}: configure FAILED — SKIPPING")
                    continue

            # Step 2: Activate (Inactive → Active)
            current_state = self._get_current_state(node_name)
            if current_state == STATE_INACTIVE:
                if self._change_state_and_verify(
                    node_name,
                    Transition.TRANSITION_ACTIVATE,
                    expected_state=STATE_ACTIVE
                ):
                    self.get_logger().info(f"  ✅ {node_name}: READY (ACTIVE)")
                else:
                    self.get_logger().error(
                        f"  ❌ {node_name}: activate FAILED — node in "
                        f"{self._state_label(self._get_current_state(node_name))}"
                    )
            elif current_state == STATE_ACTIVE:
                self.get_logger().info(f"  ✅ {node_name} already ACTIVE")
            else:
                self.get_logger().error(
                    f"  ❌ {node_name}: cannot activate from state "
                    f"{self._state_label(current_state)}"
                )

        self.get_logger().info("🏁 Lifecycle boot sequence complete")

    # ==================================================================
    # Core: change_state → poll get_state → verify
    # ==================================================================
    def _change_state_and_verify(self, node_name, transition_id, expected_state,
                                  max_retries=3):
        """Call change_state, then poll get_state until node reaches expected_state.

        Returns True only when the node is CONFIRMED in expected_state.
        """
        label = self._transition_label(transition_id)
        expected_label = self._state_label(expected_state)

        for attempt in range(1, max_retries + 1):
            self.get_logger().info(
                f"  🔄 {node_name}: {label} (attempt {attempt}/{max_retries})"
            )

            # --- Call change_state ---
            client = self.create_client(ChangeState, f"/{node_name}/change_state")
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f"  ❌ change_state service lost for {node_name}")
                return False

            request = ChangeState.Request()
            request.transition = Transition()
            request.transition.id = transition_id
            request.transition.label = label

            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout)

            if not future.done():
                self.get_logger().warn(f"  ⏱️  {label} call timed out after {self._timeout}s")
                continue

            result = future.result()
            if result is None:
                self.get_logger().warn(f"  ⚠️  {label} returned None")
                continue

            self.get_logger().info(
                f"  📨 {label} response: success={result.success}"
            )

            # --- Poll get_state until expected_state or timeout ---
            start = time.time()
            last_state = -1
            while time.time() - start < self._max_state_wait:
                current = self._get_current_state(node_name)
                if current != last_state:
                    self.get_logger().info(
                        f"  ⏳ State: {self._state_label(current)} (id={current})"
                    )
                    last_state = current

                if current == expected_state:
                    self.get_logger().info(
                        f"  ✅ {node_name}: CONFIRMED {expected_label}"
                    )
                    return True

                # Also check for error states
                if current == STATE_FINALIZED:
                    self.get_logger().error(
                        f"  ❌ {node_name} entered FINALIZED during {label}"
                    )
                    return False

                time.sleep(self._poll_interval)

            # Timeout
            self.get_logger().error(
                f"  ❌ {node_name}: {label} did not reach {expected_label} "
                f"within {self._max_state_wait}s (stuck at {self._state_label(last_state)})"
            )

        return False

    # ==================================================================
    # Helpers
    # ==================================================================
    def _get_current_state(self, node_name):
        """Query /node_name/get_state and return the primary state id."""
        client = self.create_client(GetState, f"/{node_name}/get_state")
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f"  ⚠️  get_state not available for {node_name}")
            return -1

        request = GetState.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.done() and future.result() is not None:
            return future.result().current_state.id
        return -1

    def _wait_for_service(self, node_name, service_type, timeout=30.0):
        service_name = f"/{node_name}/{service_type}"
        srv_type = ChangeState if service_type == "change_state" else GetState
        client = self.create_client(srv_type, service_name)

        self.get_logger().info(f"  Waiting for {service_name}...")
        start = self.get_clock().now()

        while not client.wait_for_service(timeout_sec=1.0):
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout:
                self.get_logger().error(
                    f"  ❌ {service_name} not available after {timeout}s"
                )
                return False

        self.get_logger().info(f"  ✅ {service_name} available")
        return True

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

    def _state_label(self, state_id):
        labels = {
            STATE_UNCONFIGURED: "UNCONFIGURED",
            STATE_INACTIVE: "INACTIVE",
            STATE_ACTIVE: "ACTIVE",
            STATE_FINALIZED: "FINALIZED",
            2: "ACTIVATING",
        }
        return labels.get(state_id, f"STATE_{state_id}")


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
