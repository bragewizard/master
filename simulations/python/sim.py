from config import NETWORK_SHAPE
import numpy as np
import threading
import random
from core import intensity_to_delay_encoding
import heapq
from collections import deque  # For spike history "memory"

# --- Synapse Constants ---
SYNAPSE_DELAY = 1.0  # 1ms delay

# --- New Model Constants ---
FIRING_THRESHOLD = 3
INTEGRATION_WINDOW = 2.0  # 5ms time window to be "co-active"
GROWTH_PROB = 0.1  # 5% chance per pre-synaptic spike
PAUSE_BETWEEN_IMAGES = 30.0  # 10ms pause, must be > INTEGRATION_WINDOW


class Simulation:
    def __init__(self, data: np.ndarray):
        print("[Sim] Initializing simulation...")
        self.now_time = 0.0
        self.iteration = 0
        self.wait_time = PAUSE_BETWEEN_IMAGES
        self.event_queue = []
        self.event_counter = 0

        self.neurons = np.zeros(NETWORK_SHAPE[0] + NETWORK_SHAPE[1])
        self.pre_to_post = {}  # {pre_idx: [post_idx, ...]}
        self.post_to_pre = {}  # {post_idx: [pre_idx, ...]}

        self.post_potential = {}
        self.post_spike_window = {}  # {post_idx: deque([(t, pre_idx), ...])}

        self.fill_event_queue(data[0], self.now_time)
        print(f"[Sim] Built event queue with {len(self.event_queue)} events.")

    def _add_synapse(self, pre_idx, post_idx):
        """
        Helper function to add a synapse (binary).
        """
        if pre_idx not in self.post_to_pre.get(post_idx, []):
            self.pre_to_post.setdefault(pre_idx, []).append(post_idx)
            self.post_to_pre.setdefault(post_idx, []).append(pre_idx)
            return (pre_idx, post_idx)
        return None

    def _remove_synapse(self, pre_idx, post_idx):
        """
        Helper function to remove a synapse (binary).
        """
        removed = False
        if post_idx in self.post_to_pre and pre_idx in self.post_to_pre[post_idx]:
            self.post_to_pre[post_idx].remove(pre_idx)
            if not self.post_to_pre[post_idx]:
                del self.post_to_pre[post_idx]
            removed = True

        if pre_idx in self.pre_to_post and post_idx in self.pre_to_post[pre_idx]:
            self.pre_to_post[pre_idx].remove(post_idx)
            if not self.pre_to_post[pre_idx]:
                del self.pre_to_post[pre_idx]
            removed = True

        if removed:
            return (pre_idx, post_idx)
        return None

    def advance(self, time_slice: float):
        target_time = self.now_time + time_slice
        newly_spiked_indices = []
        newly_grown_synapses = []
        newly_pruned_synapses = []

        is_empty = False

        # --- Corrected Pause Logic ---
        if not self.event_queue:
            if self.wait_time > 0:
                time_to_wait = min(self.wait_time, time_slice)
                self.now_time += time_to_wait
                self.wait_time -= time_to_wait
            else:
                self.wait_time = PAUSE_BETWEEN_IMAGES
                is_empty = True

            return (
                newly_spiked_indices,
                newly_grown_synapses,
                newly_pruned_synapses,
                is_empty,
            )

        while self.event_queue and self.event_queue[0][0] <= target_time:
            event_time, _counter, event_type, target_data = heapq.heappop(
                self.event_queue
            )

            self.now_time = event_time

            if event_type == "spike":
                pre_idx = target_data
                newly_spiked_indices.append(pre_idx)

                if pre_idx in self.pre_to_post:
                    for post_idx in self.pre_to_post[pre_idx]:
                        new_event_time = self.now_time + SYNAPSE_DELAY
                        new_event = (
                            new_event_time,
                            self.event_counter,
                            "arrival",
                            (pre_idx, post_idx),
                        )
                        heapq.heappush(self.event_queue, new_event)
                        self.event_counter += 1

                if random.random() < GROWTH_PROB:
                    target_post_idx = random.randint(
                        NETWORK_SHAPE[0], NETWORK_SHAPE[0] + NETWORK_SHAPE[1] - 1
                    )
                    new_synapse = self._add_synapse(pre_idx, target_post_idx)
                    if new_synapse:
                        newly_grown_synapses.append(new_synapse)

            # --- 3. "Fixed Window" Arrival ---
            elif event_type == "arrival":
                pre_idx_that_caused_it, post_idx = target_data
                arrival_time = event_time

                post_idx_window = self.post_spike_window.setdefault(post_idx, deque())

                spike_was_added = False

                if not post_idx_window:
                    post_idx_window.append((arrival_time, pre_idx_that_caused_it))
                    spike_was_added = True
                else:
                    t_start = post_idx_window[0][0]
                    t_end = t_start + INTEGRATION_WINDOW

                    if arrival_time <= t_end:
                        # Spike is INSIDE the window. Add it.
                        post_idx_window.append((arrival_time, pre_idx_that_caused_it))
                        spike_was_added = True
                    else:
                        # Spike is OUTSIDE ("too late").
                        # Per your request, NO PRUNING here.
                        # We just clear the failed window.
                        post_idx_window.clear()

                # --- 4. Firing Logic (This is now the ONLY place pruning happens) ---
                if spike_was_added:
                    current_potential = len(post_idx_window)
                    self.post_potential[post_idx] = current_potential

                    if current_potential >= FIRING_THRESHOLD:
                        # 4a. Fire!
                        newly_spiked_indices.append(post_idx)

                        # --- 4b. PRUNING ON NON-PARTICIPATION ---
                        # This is now the ONLY pruning rule.
                        contributing_pre_set = set(pre for (t, pre) in post_idx_window)
                        all_connected_pre_set = set(self.post_to_pre.get(post_idx, []))
                        non_contributing_pre_set = (
                            all_connected_pre_set - contributing_pre_set
                        )

                        for pre_to_prune in non_contributing_pre_set:
                            pruned = self._remove_synapse(pre_to_prune, post_idx)
                            if pruned:
                                newly_pruned_synapses.append(pruned)

                        # 4c. Reset after firing
                        self.post_potential[post_idx] = 0
                        post_idx_window.clear()

        if self.event_queue:
            self.now_time = target_time

        return (
            newly_spiked_indices,
            newly_grown_synapses,
            newly_pruned_synapses,
            is_empty,
        )

    def fill_event_queue(self, input_image, start_time):
        spikes = intensity_to_delay_encoding(input_image)
        height, width = spikes.shape
        y_coords, x_coords = np.where(~np.isnan(spikes))
        times = spikes[y_coords, x_coords] + start_time
        neuron_indices = y_coords * width + x_coords
        sorted_indices = np.argsort(times)
        spikes = times[sorted_indices]
        indices = neuron_indices[sorted_indices]

        for t, target in zip(spikes, indices):
            entry = (t, self.event_counter, "spike", target)
            heapq.heappush(self.event_queue, entry)
            self.event_counter += 1

        return input_image

    def next_data(self, data):
        self.iteration += 1
        if self.iteration >= len(data):
            print("[Sim] End of data, looping.")
            self.iteration = 0

            # The wait_time logic in advance() handles the pause
            # self.now_time += PAUSE_BETWEEN_IMAGES

        self.post_potential.clear()
        self.post_spike_window.clear()

        self.fill_event_queue(data[self.iteration], self.now_time)

    def init_connection(self):
        pass
