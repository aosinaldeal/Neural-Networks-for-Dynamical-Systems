import os

import torch
import numpy as np

from src.utils.plotting import (
    plot_prediction,
    plot_absolute_error,
    plot_double_pendulum_prediction,
    plot_double_pendulum_error,
    SYSTEM_TITLES,
)
from src.data.generator import generate_physical_dataset
from src import config
from src.utils.metrics import (
    calculate_metrics,
    calculate_double_pendulum_metrics,
)
from src.data.dataset import (
    numpy_to_tensor,
    split_dataset,
)
from src.utils.report import generate_report
from src.models.neural_network import NeuralNetwork
from src.utils.model_io import load_model
from experiments.summary import save_result


# ==========================================
# Evaluation
# ==========================================

def evaluate(
    experiment=None,
    experiment_name=None,
    model_name=None,
    beta=None,
    seed=None,
    delta=None
):

    if experiment is None:
        experiment = 1

    if experiment_name is None:
        experiment_name = config.EXPERIMENT_NAME

    if beta is None:
        beta = config.BETA

    if seed is None:
        seed = config.RANDOM_SEED
    
    if delta is None:
        delta = config.DUFFING_DELTA

    torch.manual_seed(seed)

    # ----------------------------------------
    # Device
    # ----------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ----------------------------------------
    # Generate data
    # ----------------------------------------

    time, position = generate_physical_dataset(
        beta=beta,
        delta=delta,
    )

    output_dim = 1 if position.ndim == 1 else position.shape[1]

    time_tensor, position_tensor = numpy_to_tensor(
        time,
        position,
    )

    # ----------------------------------------
    # Split dataset
    # ----------------------------------------

    (
        train_time,
        train_position,
        test_time,
        test_position,
    ) = split_dataset(
        time_tensor,
        position_tensor,
        train_ratio=config.TRAIN_RATIO,
        method=config.SPLIT_METHOD,
    )

    test_time = test_time.to(device)
    test_position = test_position.to(device)

    # ----------------------------------------
    # Load model
    # ----------------------------------------

    model = NeuralNetwork(
        output_dim=output_dim
    ).to(device)

    if model_name is None:
        model_name = experiment_name


    load_model(
        model,
        f"{config.TYPE}-{model_name}",
    )


    model.eval()

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    with torch.no_grad():
        predictions = model(test_time)

    # ----------------------------------------
    # Prediction
    # ----------------------------------------   

    if config.TYPE == "double_pendulum":

        (
            mse,
            mae,
            max_error,
            r2,
            theta1_mse,
            theta1_mae,
            theta1_max_error,
            theta1_r2,
            theta2_mse,
            theta2_mae,
            theta2_max_error,
            theta2_r2,
        ) = calculate_double_pendulum_metrics(
            predictions,
            test_position,
        )

    else:

        mse, mae, max_error, r2 = calculate_metrics(
            predictions,
            test_position,
        )
        
    # ========================================
    # Convert to NumPy
    # ========================================

    time_np = (
        test_time
        .cpu()
        .numpy()
        .flatten()
    )

    real_np = (
        test_position
        .detach()
        .cpu()
        .numpy()
    )

    pred_np = (
        predictions
        .detach()
        .cpu()
        .numpy()
    )

    if config.TYPE == "double_pendulum":

        # Errors

        error_theta1 = abs(
            real_np[:,0] -
            pred_np[:,0]
        )

        error_theta2 = abs(
            real_np[:,1] -
            pred_np[:,1]
        )

    # ----------------------------------------
    # Sort by time
    # ----------------------------------------

    order = time_np.argsort()

    time_np = time_np[order]
    real_np = real_np[order]
    pred_np = pred_np[order]

    # ========================================
    # Results folder
    # ========================================

    results_folder = os.path.join(
        "results",
        config.TYPE,
        config.SWEEP_NAME,
        experiment_name,
    )

    os.makedirs(
        results_folder,
        exist_ok=True,
    )

    if config.TYPE == "double_pendulum":

        plot_double_pendulum_prediction(
            time=time_np,
            reference=real_np[:,0],
            prediction=pred_np[:,0],
            theta_number=1,
            save_path=os.path.join(
                results_folder,
                "theta1_prediction.png",
            ),
        )


        plot_double_pendulum_prediction(
            time=time_np,
            reference=real_np[:,1],
            prediction=pred_np[:,1],
            theta_number=2,
            save_path=os.path.join(
                results_folder,
                "theta2_prediction.png",
            ),
        )


        plot_double_pendulum_error(
            time=time_np,
            error=error_theta1,
            theta_number=1,
            save_path=os.path.join(
                results_folder,
                "theta1_error.png",
            ),
        )


        plot_double_pendulum_error(
            time=time_np,
            error=error_theta2,
            theta_number=2,
            save_path=os.path.join(
                results_folder,
                "theta2_error.png",
            ),
        )

    else:

        plot_prediction(
            time=time_np,
            reference=real_np,
            prediction=pred_np,
            title=SYSTEM_TITLES.get(
                config.TYPE,
                "Oscillator Motion Approximation",
            ),
            xlabel="Time (s)",
            ylabel="θ(t) (rad)",
            save_path=os.path.join(
                results_folder,
                f"{config.TYPE}_prediction.png",
            ),
        )
    


    if config.TYPE != "double_pendulum":

        # ========================================
        # Absolute error plot
        # ========================================

        absolute_error = abs(
            real_np - pred_np
        )

        plot_absolute_error(
            time=time_np,
            error=absolute_error,
            save_path=os.path.join(
                results_folder,
                "absolute_error.png",
            ),
        )

    

    # ========================================
    # Save result to summary
    # ========================================
    metrics_path = os.path.join(
        results_folder,
        "metrics.txt",
    )

    metrics = {
        "mse": mse.item(),
        "mae": mae.item(),
        "max_error": max_error.item(),
        "r2": r2.item(),
    }


    if config.TYPE == "double_pendulum":

        metrics.update(
            {
                "theta1_mse": theta1_mse.item(),
                "theta1_mae": theta1_mae.item(),
                "theta1_max_error": theta1_max_error.item(),
                "theta1_r2": theta1_r2.item(),

                "theta2_mse": theta2_mse.item(),
                "theta2_mae": theta2_mae.item(),
                "theta2_max_error": theta2_max_error.item(),
                "theta2_r2": theta2_r2.item(),
            }
        )


    generate_report(
        experiment_name=experiment_name,
        metrics=metrics,
        save_path=metrics_path,
    )

    save_result(
        experiment=experiment,
        seed=seed,
        mse=mse.item(),
        mae=mae.item(),
        max_error=max_error.item(),
        r2=r2.item(),
    )



if __name__ == "__main__":
    evaluate()