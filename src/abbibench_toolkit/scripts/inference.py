from abbibench_toolkit.config.config import Config
from configs import _CONFIGS
import tyro

def inference(config: Config):
    if not config.is_correlation_only:
        # 1. load dataset and model
        dataset, model = config.data_config.dataset, config.model_config.model
        # 2. compute and save log_likelihood
        log_likelihood = model.get_log_likelihood(dataset)
        config.save_result(log_likelihood)
    else:
        log_likelihood = config.result

    # 3. compute correlation
    rho, p_value = config.data_config.compute_correlation(log_likelihood)
    print(f"rho: {rho:.4f}, p-value: {p_value:.4e}")
    return

def entrypoint():
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    inference(config)

if __name__=="__main__":
    entrypoint()
