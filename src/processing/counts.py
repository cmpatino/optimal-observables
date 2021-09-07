from typing import List

from src.processing import event_selection


def n_particles_from_tag(events, key: str) -> List[int]:
    """Calculate number of particles in each event based on boolean tags
    recorded on a TTree.

    :param events: Delphes event TTree
    :type events: TTree
    :param key: Key for TBranchElement that contains
                the boolean tag (e.g., 'Jet.BTag')
    :type key: str
    :return: Number of particles in each event.
    :rtype: List[int]
    """
    particle_count = []
    for event in events[key].array():
        particle_count.append(sum(event))
    return particle_count


def n_particles(events, key: str) -> List[int]:
    """Calculate number of particles in each event based on the length of
    a recorded quantity on a TTree.

    :param events: Delphes event TTree
    :type events: TTree
    :param key: Key for TBranchElement that records particle quantity
                to use as proxy for number of particles (e.g., 'Jet.PT')
    :type key: str
    :return: Number of particles in each event.
    :rtype: List[int]
    """
    particle = key.split(".")[0].lower()
    mask = getattr(event_selection, f"select_{particle}")(events)
    particle_count = []
    key_events = events[key].array()[mask]
    for event in key_events:
        particle_count.append(len(event))
    return particle_count
