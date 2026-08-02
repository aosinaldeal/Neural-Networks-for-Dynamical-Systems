import torch


def calculate_metrics(predictions, targets):

    mse = torch.mean(
        (predictions - targets) ** 2
    )

    mae = torch.mean(
        torch.abs(predictions - targets)
    )

    max_error = torch.max(
        torch.abs(predictions - targets)
    )

    ss_res = torch.sum(
        (targets - predictions) ** 2
    )

    ss_tot = torch.sum(
        (targets - torch.mean(targets)) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    return mse, mae, max_error, r2

def calculate_double_pendulum_metrics(
    predictions,
    targets,
):

    # Theta 1

    theta1_pred = predictions[:, 0]
    theta1_true = targets[:, 0]

    theta1_mse = torch.mean(
        (theta1_pred - theta1_true) ** 2
    )

    theta1_mae = torch.mean(
        torch.abs(theta1_pred - theta1_true)
    )

    theta1_max_error = torch.max(
        torch.abs(theta1_pred - theta1_true)
    )

    theta1_ss_res = torch.sum(
        (theta1_true - theta1_pred) ** 2
    )

    theta1_ss_tot = torch.sum(
        (theta1_true - torch.mean(theta1_true)) ** 2
    )

    theta1_r2 = (
        1 - theta1_ss_res / theta1_ss_tot
    )


    # Theta 2

    theta2_pred = predictions[:, 1]
    theta2_true = targets[:, 1]


    theta2_mse = torch.mean(
        (theta2_pred - theta2_true) ** 2
    )

    theta2_mae = torch.mean(
        torch.abs(theta2_pred - theta2_true)
    )

    theta2_max_error = torch.max(
        torch.abs(theta2_pred - theta2_true)
    )

    theta2_ss_res = torch.sum(
        (theta2_true - theta2_pred) ** 2
    )

    theta2_ss_tot = torch.sum(
        (theta2_true - torch.mean(theta2_true)) ** 2
    )

    theta2_r2 = (
        1 - theta2_ss_res / theta2_ss_tot
    )


    # Global metrics

    mse = (
        theta1_mse + theta2_mse
    ) / 2

    mae = (
        theta1_mae + theta2_mae
    ) / 2

    max_error = torch.max(
        torch.stack(
            [
                theta1_max_error,
                theta2_max_error,
            ]
        )
    )

    r2 = (
        theta1_r2 + theta2_r2
    ) / 2


    return (
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
    )