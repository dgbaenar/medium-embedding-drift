import keras
import mlflow


class MLFlowTracker:

    def __init__(self, experiment_name):
        super().__init__()
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        self._callback = MLFlowTrainTrackingCallback()

    def track_config(self, configs):
        mlflow.log_params(configs)

    def track_artifacts(self, filepath):
        mlflow.log_artifact(filepath)

    def finish_run(self):
        mlflow.end_run()

    def get_callback(self):
        return self._callback


class MLFlowTrainTrackingCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        res = {}
        for k in logs.keys():
            res[k] = logs[k]
        mlflow.log_metrics(res, step=epoch)
