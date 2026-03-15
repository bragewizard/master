import pyglet
from _gui import SNNVisualizer
from _snn import SpikingNet
from _data import LineVideoGenerator, ShapeVideoGenerator

# 1. Setup Simulation
snn = SpikingNet()
gen = ShapeVideoGenerator()

# 2. Setup Diagnostic Dashboard
viz = SNNVisualizer(snn)


def update(dt):
    # Get physics frame
    frame, _ = gen.get_next_frame()

    # Process SNN logic
    snn.set_input_currents(frame)
    all_spikes = []
    # Run 5 integration steps per frame to check for potential buildup
    for _ in range(5):
        spikes, _ = snn.advance(dt=1.0)
        all_spikes.extend(spikes)

    # Sync visual panes
    viz.update_frame(frame, all_spikes)


pyglet.clock.schedule_interval(update, 1 / 30.0)
pyglet.app.run()
