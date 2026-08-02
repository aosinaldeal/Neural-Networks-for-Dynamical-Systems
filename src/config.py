# ========================================
# Training
# ========================================

EPOCHS = 500
LEARNING_RATE = 0.0005
BATCH_SIZE = 32

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ========================================
# Neural Network
# ========================================

HIDDEN_LAYERS = 3
NEURONS_PER_LAYER = 128
ACTIVATION = "Tanh"


# ========================================
# Dataset
# ========================================

DURATION = 30
SAMPLES = 6000

# ========================================
# System selection
# ========================================

TYPE = "pendulum"

    # Available:
    # harmonic
    # damped_harmonic
    # forced_harmonic
    # duffing
    # pendulum
    # double_pendulum


SWEEP_NAME = "test"
EXPERIMENT_NAME = "experiment_21"
SPLIT_METHOD = "random"

# ========================================
# Oscillators
# ========================================

AMPLITUDE = 1
OMEGA = 2
BETA = 1
PHASE = 0


FORCING_AMPLITUDE = 0.5
FORCING_OMEGA = 1.8


# ========================================
# Duffing
# ========================================

DUFFING_DELTA = 0.2
DUFFING_ALPHA = 1.0
DUFFING_BETA = 1.0
DUFFING_GAMMA = 0.3
DUFFING_OMEGA = 1.2

INITIAL_POSITION = 1.0
INITIAL_VELOCITY = 0.0


# ========================================
# Pendulum
# ========================================

PENDULUM_GRAVITY = 9.81
PENDULUM_LENGTH = 1.0

INITIAL_ANGLE = 1.0
INITIAL_ANGULAR_VELOCITY = 0.0


# ========================================
# Double Pendulum
# ========================================

DOUBLE_PENDULUM_MASS_1 = 1.0
DOUBLE_PENDULUM_MASS_2 = 1.0

DOUBLE_PENDULUM_LENGTH_1 = 1.0
DOUBLE_PENDULUM_LENGTH_2 = 1.0

DOUBLE_PENDULUM_GRAVITY = 9.81

INITIAL_THETA_1 = 1.0
INITIAL_THETA_2 = 1.0

INITIAL_OMEGA_1 = 0.0
INITIAL_OMEGA_2 = 0.0