import evidently
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import model
from evidently import ColumnMapping
from evidently.metric_preset import TargetDriftPreset


class MultiClassVisionModelDriftMonitor:
    def __init__(self, index_to_class, small_subset=None, big_subset=None):
        self.column_mapping = ColumnMapping()
        self.column_mapping.target_names = index_to_class
        self.column_mapping.embeddings = {
            'small_subset': small_subset,
            'big_subset': big_subset
        }
    
    def run_target_drift_report(self, reference_data, production_data, target_col, prediction_col):
        # Update column mapping for target and prediction
        self.column_mapping.target = target_col
        self.column_mapping.prediction = prediction_col
        
        # Setup and run target drift report
        target_drift_report = Report(metrics=[
            TargetDriftPreset(stattest='jensenshannon'),
        ])
        target_drift_report.run(reference_data=reference_data, current_data=production_data, column_mapping=self.column_mapping)
        return target_drift_report
    
    def run_embeddings_drift_report(self, subset_size, reference_data, production_data):
        # Setup and run embeddings drift report
        embeddings_drift_report = Report(metrics=[
            EmbeddingsDriftMetric(subset_size, drift_method=model(threshold=0.55, bootstrap = None,
                              pca_components = 20,)),
        ])
        embeddings_drift_report.run(reference_data=reference_data, current_data=production_data, column_mapping=self.column_mapping)
        return embeddings_drift_report
