# Weights (not included)

Place the Mammo-CLIP EfficientNet-B2 image-encoder checkpoint here:

    weights/mammoclip/b2-model-best-epoch-10.tar

Download it from the official Mammo-CLIP repository (https://github.com/batmanlab/Mammo-CLIP), following its
instructions and licence. The training script opens it with a restricted unpickler (`c1_concepts/safe_ckpt.py`)
that resolves only allowlisted classes.
