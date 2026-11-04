"""Stage 2: pretrained VLA policies and their adapters.

``smolvla`` needs the ``vla`` extra (torch + LeRobot); the adapter and the policy
source do not, so they are importable and testable without a GPU.
"""

from so101_vla.policies.action_adapter import ActionChunkValidator, ChunkReport, InvalidChunk
from so101_vla.policies.policy_source import PolicySource

__all__ = ["ActionChunkValidator", "ChunkReport", "InvalidChunk", "PolicySource"]
