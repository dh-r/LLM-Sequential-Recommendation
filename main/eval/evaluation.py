import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Tuple, Optional, Literal
import matplotlib.pyplot as plt
import numpy as np
import time

import pandas as pd
from multiprocess import Pool
from multiprocess.pool import AsyncResult
from pandas.io.formats.style import Styler

from main.data.abstract_dataset import Dataset
from main.eval import metrics
from main.eval.metrics.metric import RankingMetric, MetricDependency
from main.abstract_model import Model

logging.basicConfig()
logger = logging.getLogger()


@dataclass
class EvaluationReport:
    """A report of the evaluation of a recommender model.

    Args:
        model_name (str): The name of the model evaluated.
        top_k (int): The top-k predictions considered for this evaluation.
        results: (Dict[str, Any]): The (averaged) results per evaluation metric.
        results_per_sample: (Dict[str, np.ndarray]): The results per evaluation metric
            per user. This is not stored by default.
    """

    model_name: str
    top_k: int
    results: Dict[str, Any] = field(default_factory=lambda: {})
    results_per_sample: Dict[Any, np.ndarray] = field(default_factory=lambda: {})

    def to_df(self) -> pd.DataFrame:
        """Transforms report to pandas DF.

        Returns:
            pd.DataFrame: The global evaluation results.
        """
        df = pd.DataFrame.from_dict(self.results.items()).transpose()
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        df.index = [self.model_name]
        return df

    def to_json(self) -> str:
        """Returns the reports as json string.

        Returns:
            str: The report as a string.
        """
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def average(reports: List["EvaluationReport"]) -> Optional["EvaluationReport"]:
        """Average the results from a set of evaluation reports.
        Useful when averaging across multiple folds.

        Note: Does not support average results per sample.

        Args:
            reports (List[EvaluationReport]): The list of evaluation reports.

        Returns:
            EvaluationReport: A single averaged evaluation report.
        """
        if len(reports) == 0:
            return None

        metric_names = reports[0].results.keys()
        new_metrics = {}

        # Average metrics across reports.
        for metric_name in metric_names:
            sum_metric = 0
            for report in reports:
                sum_metric += report.results[metric_name]
            new_metrics[metric_name] = sum_metric / len(reports)

        for report in reports:
            if len(report.results_per_sample) > 0:
                logging.warning(
                    """Average results per sample is currently not supported.
                    Sample results per user will be omitted."""
                )

        return EvaluationReport(reports[0].model_name, reports[0].top_k, new_metrics)

    def __repr__(self) -> str:
        """A representation of this report.

        Returns:
            str: A string representation of the evaluation report.
        """
        return f"""Evaluation report for {self.model_name} with
                metrics {', '.join(list(self.results.keys()))} and
                results_per_sample={len(self.results_per_sample) > 0}."""


class Evaluation:
    """Implementation for the evaluation of top-k recommender systems.

    An evaluation can be run in two ways:
    1. Directly evaluate against the predictions and ground truths:
        `Evaluation.eval(predictions, ground_truths, top_k=5)`
    2. Run and evaluate a single or multiple models and a dataset:
        `evaluation = Evaluation(model=[item_cf, popularity], dataset=sparse,
                                    use_folds=True)`
        `evaluation.run(top_k=5)`
    This allows the metric to directly apply cross-validation/testing, if necessary.

    Both implementations support multiprocessing by dividing the
    predictions/ground-truth combinations across the specified cores.
    """

    def __init__(
        self,
        models: Union[Model, List[Model]],
        dataset: Optional[Dataset],
        use_folds: bool = False,
        is_verbose: bool = False,
    ):
        """Initializes a new evaluation for a single or multiple models given a dataset."""

        # Check if it folds are available.
        if dataset is not None and use_folds and not dataset.has_k_fold():
            raise Exception(
                """Can't run evaluation with folds on a dataset which does not have any
                folds. Please ensure the dataset has the folds computed."""
            )

        # Set log level based on verbosity.
        if is_verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        if not isinstance(models, List):
            models = [models]

        self.models: List[Model] = models
        self.dataset: Optional[Dataset] = dataset
        self.use_folds: bool = use_folds

        self.reports: List[EvaluationReport] = []

    def run(
        self,
        top_k: int = 10,
        metrics: List[RankingMetric] = metrics.ALL_DEFAULT,
        metrics_per_sample: bool = False,
        dependencies: Dict[MetricDependency, Any] = {},
        cores: int = 1,
    ) -> Union[EvaluationReport, List[EvaluationReport]]:
        """Runs the evaluation on all the given models. First the models are trained
        with folds or with the train data. Then the model predicts the samples which
        are in the test set or validation fold. Finally, the predictions and
        ground-truths are evaluated. Whenever self.use_folds is enabled, results are
        averaged across folds.

        Args:
            top_k (int, optional): The cut-off point for the recommendations.
                Defaults to 10.
            metrics (List[RankingMetric], optional): Metrics to use for evaluation.
                Defaults to metrics.ALL_DEFAULT.
            metrics_per_sample (bool, optional): If enabled, returns the metric
                result per sample. A sample is a single user, session or interaction.
                Defaults to False.
            dependencies (Dict[MetricDependency, Any], optional): Specifies the
                dependencies for this evaluation run. Some metrics require external
                dependencies, such as `item counts`, to compute the result. This dict
                can be used to specify certain dependencies. If not given, but
                required by a metric an exception will be thrown. In some scenarios,
                we are able to derive and ingest dependencies automatically.
            cores (int, optional): Compute this evaluation across multiple cores.
                Defaults to 1.
        Returns:
            Union[EvaluationReport, List[EvaluationReport]]: Either a single or multiple
                evaluation reports.
        """
        if self.dataset is None:
            raise Exception(
                """Can't run evaluation without a dataset. Please construct this
                instance with a dataset attribute."""
            )
        if metrics_per_sample and self.use_folds:
            raise Exception(
                """Currently the use of metrics_per_sample and using folds is not
                supported. Please disable one of these options."""
            )

        # Clear current reports.
        self.reports.clear()

        if self.use_folds:
            total_folds = len(self.dataset.get_k_fold_eval())
            fold_reports: List[EvaluationReport] = []

            for model in self.models:
                model_name = type(model).__name__
                logger.info(
                    f"""[EVALUATION] Now starting cross-validation runs for {model_name}
                    with {total_folds} folds."""
                )
                for i, (train_data, test_data_labels, test_data) in enumerate(
                    self.dataset.get_k_fold_eval()
                ):
                    logger.info(
                        f"""[EVALUATION] [{i + 1}/{total_folds}] Training {model_name}
                        started."""
                    )
                    model.train(train_data)
                    logger.info(
                        f"""[EVALUATION] [{i + 1}/{total_folds}] Training {model_name}
                        completed."""
                    )

                    logger.info(
                        f"""[EVALUATION] [{i + 1}/{total_folds}] Prediction {model_name}
                        started."""
                    )

                    recommendations = model.predict(test_data_labels, top_k=top_k)
                    logger.info(
                        f"""[EVALUATION] [{i + 1}/{total_folds}] Prediction {model_name}
                        completed."""
                    )

                    report = Evaluation.eval(
                        recommendations,
                        test_data,
                        top_k,
                        metrics,
                        metrics_per_sample,
                        {
                            MetricDependency.NUM_ITEMS: self.dataset.get_unique_item_count(),
                            MetricDependency.ITEM_COUNT: self.dataset.get_item_counts(),
                            MetricDependency.SAMPLE_COUNT: self.dataset.get_sample_counts(),
                        }
                        | dependencies,
                        cores,
                        model_name,
                    )

                    fold_reports.append(report)
                self.reports.append(EvaluationReport.average(fold_reports))
        else:  # When use_folds is disabled, we use the train and test set.
            for model in self.models:
                model_name = model.name()
                train_data = self.dataset.train_data
                test_data_labels, test_data = self.dataset.get_test_data_eval()

                logger.info(f"[EVALUATION] Training {model_name} started.")
                start_time = time.perf_counter()
                model.train(train_data)
                end_time = time.perf_counter()

                logger.info(
                    f"[EVALUATION] Training {model_name} completed. That took {end_time - start_time} seconds."
                )

                logger.info(f"[EVALUATION] Prediction {model_name} started.")
                start_time = time.perf_counter()
                recommendations = model.predict(test_data_labels, top_k=top_k)
                end_time = time.perf_counter()
                logger.info(
                    f"[EVALUATION] Prediction {model_name} completed. That took {end_time - start_time} seconds."
                )

                report = Evaluation.eval(
                    recommendations,
                    test_data,
                    top_k,
                    metrics,
                    metrics_per_sample,
                    {
                        MetricDependency.NUM_ITEMS: self.dataset.get_unique_item_count(),
                        MetricDependency.ITEM_COUNT: self.dataset.get_item_counts(),
                        MetricDependency.SAMPLE_COUNT: self.dataset.get_sample_counts(),
                    }
                    | dependencies,
                    cores,
                    model_name,
                )

                self.reports.append(report)

        # For convenience, we send a list or a single instance.
        if len(self.reports) > 1:
            return self.reports

        return self.reports[0]

    @staticmethod
    def count_unique_items(all_items: List[np.ndarray]) -> int:
        """Computes the unique values in a list of numpy arrays.
        This method is used to compute the unique item space.

        Args:
            all_items (List[np.ndarray]): A list of all items.

        Returns:
            int: Unique item count.
        """
        return len(np.unique(np.concatenate(all_items)))

    def results_as_table(
        self, caption: str = "", max_color: str = "lightgreen"
    ) -> Optional[Styler]:
        """Summarizes all results a single table.

        Args:
            caption (str, optional): The caption of the table. Defaults to "".
            max_color: A string representing the color of the max cells of the results
                table.

        Returns:
            pd.DataFrame: The table as a pandas dataframe.
        """
        if len(self.reports) == 0:
            raise Exception(
                """[EVALUATION] No reports have been generated.
                Please use run() first."""
            )

        # Ensure unique model names.
        model_names = set()
        for report in self.reports:
            if report.model_name in model_names:
                report.model_name = report.model_name + "_" + str(len(model_names))
            model_names.add(report.model_name)

        eval_reports = [report.to_df() for report in self.reports]
        all_results = pd.concat(eval_reports)
        all_results.index.name = None
        all_results.columns.name = None

        all_results = all_results.style.highlight_max(
            color=max_color,
            subset=list(self.reports[0].results.keys()),
        )
        all_results.set_caption(caption)
        return all_results

    def plot_results_per_sample(self, metric_to_plot="Precision"):
        """Plots the results of a single metric for all samples.
        The results are sorted in descending order.

        Args:
            metric_to_plot (str, optional): The metric to plot. Defaults to "Precision".

        Returns:
            A matplotlib plot.
        """
        plt.rcParams["figure.figsize"] = [7, 3]
        plt.rcParams["figure.dpi"] = 200

        for i, report in enumerate(self.reports):
            for j, (metric, sample_results) in enumerate(
                report.results_per_sample.items()
            ):
                if not (
                    metric.rstrip(f"@{report.top_k}") == metric_to_plot
                    or metric == metric_to_plot
                ):
                    continue

                metric_sorted = np.sort(sample_results)
                plt.plot(
                    metric_sorted[::-1],
                    label=f"{report.model_name} - {metric}",
                )

        plt.legend()
        return plt

    @staticmethod
    def eval(
        predictions: Dict[int, np.ndarray],
        ground_truths: Dict[int, np.ndarray],
        top_k: int = 10,
        metrics: List[RankingMetric] = metrics.ALL_DEFAULT,
        metrics_per_sample: bool = False,
        dependencies: Dict[MetricDependency, Any] = {},
        cores: int = 1,
        model_name: str = "model",
    ) -> EvaluationReport:
        """Evaluates a set of predictions and ground-truths.

        Args:
            predictions (Dict[int, np.ndarray]): The predictions in a dictionary form,
                where each key is the sample identifier (int) and the value is an
                ordered list of predicted item ids. Complies with the output
                of each model.
            ground_truths (Dict[int, np.ndarray]): The ground-truths in a
                dictionary form, where each key is the sample identifier (int) and the
                value is an ordered list of predicted item ids.
            top_k (int, optional): The cut-off point for the recommendations. It is
                not required that `predictions` has <= top_k predictions per sample.
                This method will only evaluate against the top_k predictions.
                Defaults to 10.
            metrics (List[RankingMetric], optional): Metrics to use for evaluation.
                Defaults to metrics.ALL_DEFAULT.
            metrics_per_sample (bool, optional): If enabled, returns the metric
                result per sample. A sample is a single user, session or interaction.
                Defaults to False.
            dependencies (Dict[MetricDependency, Any], optional): Specifies the
                dependencies for this evaluation run. Some metrics require external
                dependencies, such as `item counts`, to compute the result. This dict
                can be used to specify certain dependencies. If not given, but
                required by a metric an exception will be thrown. In some scenarios,
                we are able to derive and ingest dependencies automatically.
            cores (int, optional): Compute this evaluation across multiple cores.
                Defaults to 1.
            model_name (str, optional): The name of the model or evaluation.
                Defaults to "model".

        Returns:
            EvaluationReport: A report on the evaluation results.
        """
        # Prepare evaluation, ensures predictions and ground_truth are of equal length,
        # and that there are only top-k predictions.
        (
            predictions,
            ground_truths,
            intersect,
            sample_ids,
        ) = Evaluation.prepare_evaluation(predictions, ground_truths, top_k)

        logger.info(f"[EVALUATION] Running evaluation on {cores} core(s).")

        # Compute data characteristics for metrics.
        # Each metric gets 'num_items', 'num_samples', 'top_k' by default.
        if MetricDependency.NUM_ITEMS not in dependencies:
            # We assume the union of test and predict data covers the item-space.
            # We need to know this for metrics such as catalog coverage.
            # It is good to keep in mind this is an approximation of the total items.
            logging.warning(
                f"NUM_ITEMS was not explicitly set, now deriving it from predictions and"
                f"ground truths."
            )
            num_items = Evaluation.count_unique_items(predictions + ground_truths)
        else:
            num_items = dependencies[MetricDependency.NUM_ITEMS]
        num_samples = len(predictions)

        # Initialize the metrics.
        metrics_init: List[RankingMetric] = []
        for m_class in metrics:
            # Manage dependencies for a metric.
            metric: RankingMetric = m_class.__class__()
            metric.set_top_k(top_k)
            metric.set_num_items(num_items)
            metric.set_num_samples(num_samples)

            # Check if we have the required dependency for this metric.
            deps_for_this_metric = {}
            for dep in metric.get_required_dependencies():
                if dep not in dependencies:
                    raise AttributeError(
                        f"{metric.name()} has a {dep} dependency."
                        f"Please pass it in the 'dependencies' argument."
                    )
                deps_for_this_metric[dep] = dependencies[dep]
            metric.set_dependencies(deps_for_this_metric)

            metrics_init.append(metric)

        metrics = metrics_init

        eval_report = EvaluationReport(model_name, top_k)

        # We are not going to distribute it across cores.
        if cores == 1:
            for metric in metrics:
                logger.info(f"[EVALUATION] Computing {metric.name()} started.")

                metric_state = metric.state_init()
                metric_result = metric.eval_partial(
                    predictions,
                    ground_truths,
                    intersect,
                    sample_ids,
                    metrics_per_sample,
                )

                # Unpack tuple if we have metrics per sample.
                if metrics_per_sample:
                    metric_result, metric_result_bulk = metric_result
                    eval_report.results_per_sample[metric.name()] = np.array(
                        metric_result_bulk
                    )

                # Always merge and finalize, to average results across samples.
                metric_result = metric.state_merge(metric_state, metric_result)
                eval_report.results[metric.name()] = metric.state_finalize(
                    metric_result
                )

                logger.info(f"[EVALUATION] Computing {metric.name()} completed.")
            return eval_report

        logger.info(
            f"[EVALUATION] Computing {', '.join([m.name() for m in metrics])} started."
        )

        # Prepare multi-processing.
        pool = Pool(processes=cores)
        q, r = divmod(len(predictions), cores)
        start = 0

        core_results: List[AsyncResult] = []

        # Asynchronously evaluate work across cores.
        for i in range(cores):
            size = q + (i < r)
            end = start + size

            # Evaluate metric partially on a subset of the data.
            core_results.append(
                pool.apply_async(
                    Evaluation._eval_metric,
                    [
                        predictions[start:end],
                        ground_truths[start:end],
                        intersect[start:end],
                        sample_ids[start:end],
                        metrics,
                        metrics_per_sample,
                    ],
                )
            )
            start = end

        # Get results across cores.
        core_results = [results.get() for results in core_results]

        # Go over all metrics and merge results per core.
        for i, metric in enumerate(metrics):
            metric_state = metric.state_init()
            metric_all_sample: List[Any] = []

            # Get metric results and merge results across cores.
            for all_metric_results in core_results:
                metric_result = all_metric_results[i]

                if metrics_per_sample:
                    metric_result, metric_per_sample = metric_result
                    metric_all_sample.append(metric_per_sample)

                metric_state = metric.state_merge(metric_state, metric_result)

            # Always merge and finalize, to average results across samples.
            eval_report.results[metric.name()] = metric.state_finalize(metric_state)
            if metric.per_sample() and len(metric_all_sample) > 0:
                eval_report.results_per_sample[metric.name()] = np.concatenate(
                    metric_all_sample
                )
        logger.info(
            f"[EVALUATION] Computing {', '.join([m.name() for m in metrics])} "
            f"completed."
        )

        # Close multiprocessing pool.
        pool.close()

        return eval_report

    @staticmethod
    def _eval_metric(
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        intersect: List[np.ndarray],
        sample_ids: np.ndarray,
        metrics: List[metrics.RankingMetric],
        metric_per_sample: bool,
    ) -> Union[List[Any], List[Tuple[Any, Any]]]:
        """Evaluate a set of metrics on a subset of the data.
        Since we assume this method is executed on a subset of the data, metric results
        are accumulated and not (yet) averaged.

        Args:
            predictions (List[np.ndarray]): A list of item_ids predictions.
            ground_truths (List[np.ndarray]): A list of ground_truths. Should have the
                same order as the predictions.
            intersect (List[np.ndarray]): The list of the intersection between
                predictions and ground-truths.
            sample_ids (np.ndarray): A list of sample ids corresponding to the
                predictions, ground-truths and intersect.
            metrics (List[metrics.RankingMetric]): The metrics to evaluate.
            metric_per_sample (bool): If enabled, returns the metric result per sample.

        Returns:
            Union[List[Any], List[Tuple[Any, Any]]]: Metric results.
        """
        results_per_metric = []
        for metric in metrics:
            metric_result = metric.eval_partial(
                predictions=predictions,
                ground_truths=ground_truths,
                intersect=intersect,
                sample_ids=sample_ids,
                metrics_per_sample=metric_per_sample,
            )

            results_per_metric.append(metric_result)

        return results_per_metric

    @staticmethod
    def prepare_evaluation(
        predictions: Dict[int, np.ndarray],
        ground_truths: Dict[int, np.ndarray],
        top_k: int = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Prepares predictions and ground-truth samples for evaluation:

        1. For each sample_id (the keys in the dictionaries),
            computes the intersection between the predictions and ground-truths.
        2. Transforms the dictionaries to lists of np.ndarrays.
            Within these lists the order is preserved between predictions and
            ground-truths. This ensures that each index in these lists corresponds the
            same prediction or ground-truth sample.
        3. Computes a lookup table to transform from the original sample_id to the
            location in the list.
        4. Truncates the predictions to only keep top-k recommendations (optional).

        Args:
            predictions (Dict[int, np.ndarray]): the predictions (nd.array)
                per sample_id (int).
            ground_truths (Dict[int, np.ndarray]): the ground_truths (nd.array)
                per sample_id (int).
            top_k: (int): the top_k predictions to keep.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
                a list of predictions, a list of ground-truths, a list of overlap and
                a lookup table. The order is the same for all these lists.

        """

        sample_ids: List[int] = []
        intersect_list: List[np.ndarray] = []
        predictions_list: List[np.ndarray] = []
        ground_truths_list: List[np.ndarray] = []

        # Keep track for how many samples there are less than top_k predictions.
        less_than_k_count: int = 0

        for sample_id, sample_ground_truth in ground_truths.items():
            if (
                sample_id not in predictions
            ):  # Check if sample exists in prediction data.
                logger.warning(
                    f"""[EVALUATION] Couldn't find prediction which is part of the
                    ground truths, id={sample_id}."""
                )
                continue

            if top_k is not None:
                sample_predictions = predictions[sample_id][:top_k]

                # Remove duplicates from the predictions.
                sample_predictions_seen = set()
                sample_predictions = [
                    sample_prediction
                    for sample_prediction in sample_predictions
                    if not (
                        sample_prediction in sample_predictions_seen
                        or sample_predictions_seen.add(sample_prediction)
                    )
                ]

                if len(sample_predictions) < top_k:
                    less_than_k_count += 1
            else:
                sample_predictions = predictions[sample_id]

            # Compute intersection.
            overlap = np.array(
                list(set(sample_predictions).intersection(set(sample_ground_truth)))
            )

            # All lists have the same order.
            intersect_list.append(overlap)
            predictions_list.append(sample_predictions)
            ground_truths_list.append(sample_ground_truth)
            sample_ids.append(sample_id)

        if less_than_k_count > 0:
            logger.warning(
                f"""[EVALUATION] For {less_than_k_count}/{len(ground_truths_list)}
                ground-truth samples there were less than {top_k} predictions."""
            )

        return (
            predictions_list,
            ground_truths_list,
            intersect_list,
            np.array(sample_ids),
        )

    @staticmethod
    def from_results(reports: List[EvaluationReport]) -> "Evaluation":
        """Utility method to recreate an Evaluation instance from a set of reports.
        This way one can use helper methods of the evaluation, for example, to create
        a table with all results.

        Args:
            reports (List[EvaluationReport]): A list of reports.

        Returns:
            Evaluation: An evaluation instance.
        """
        evaluation = Evaluation([], None)
        evaluation.reports = reports

        return evaluation

    @staticmethod
    def sampled_eval(
        predictions: Dict[int, np.ndarray],
        ground_truths: Dict[int, np.ndarray],
        top_l: int = 100,
        top_k: int = 10,
        num_to_sample: int = 100,
        sampling_approach: Literal["random", "popular"] = "popular",
        dependencies: Dict[MetricDependency, Any] = {},
        cores: int = 1,
        model_name: str = "model",
    ) -> EvaluationReport:
        """Evaluates a set of predictions and ground-truths with the sampling approach
        usually done in academic papers. For an example, refer to section 4.2 of the
        [original BERT4Rec paper](https://arxiv.org/pdf/1904.06690.pdf).

        Most papers randomly sample 100 items, including the ground-truth item,
        and make the model at hand rank these. The sampling method varies across papers;
        but is most often either random or popularity-based. To compare our models
        with the models in these papers, we implement the same evaluation approach.

        Our Model API does not support returning scores, so it is not possible to
        directly make the model rank the sampled items. Instead, we use the
        TOP-L items out of all items, remove the items that were not sampled, and then
        compute the metrics on the the TOP-K recommendation slate. Note that this
        approximates the approach in the literature, and serves as a lower bound.
        To clarify, this means that the score of a metric in our approximation is at
        most the score of a metric if we were to simply rank the sampled items like
        in the literature. The latter approach would have been intrusive to our models
        (by setting the scores of non-sampled items to -np.inf), or would require us
        to return all scores for all items (leading to an OOM).

        Our approach is a lower bound because of the (quite unlikely) case that the
        ground-truth item is not in the TOP-L recommendation slate, but would be in the
        TOP-K slate if we were to do it like the literature. In all other cases,
        the results would be the same.

        Args:
            predictions (Dict[int, np.ndarray]): The predictions in a dictionary form,
                where each key is the sample identifier (int) and the value is an
                ordered list of predicted item ids. Complies with the output
                of each model.
            ground_truths (Dict[int, np.ndarray]): The ground-truths in a
                dictionary form, where each key is the sample identifier (int) and the
                value is an ordered list of predicted item ids.
            top_k (int, optional): The cut-off point for the recommendations only
                including the sampled items. Defaults to 10.
            num_to_sample (int, optional): The number of negative items to sample.
            sampling_approach ("random" or "popular"): The sampling approach.
            dependencies (Dict[MetricDependency, Any], optional): Specifies the
                dependencies for this evaluation run. eval() requires this argument,
                and this method needs the item counts for sampling according to
                popularity.
            evaluation arguments: The args and kwargs for the call to eval(), excluding
                predictions, ground-truths and top_k.

        Returns:
            EvaluationReport: A report on the sampled evaluation results.
        """
        # Do some assertions about top_l and top_k.

        item_id_to_popularity = dependencies[metrics.MetricDependency.ITEM_COUNT]
        all_items = list(item_id_to_popularity.keys())
        all_item_popularities = np.array(list(item_id_to_popularity.values()))
        all_item_popularities = all_item_popularities / np.sum(all_item_popularities)

        sampled_predictions = {}

        # Create the sampled predictions.
        less_than_l_count = 0
        less_than_k_count = 0
        for session_id, session_ground_truth in ground_truths.items():
            # Assert we used leave-one-out evaluation.
            assert len(session_ground_truth) == 1

            session_ground_truth = session_ground_truth[0]
            session_predictions = predictions[session_id][:top_l]

            if len(session_predictions) < top_l:
                less_than_l_count += 1

            # Sample negative items.
            if sampling_approach == "popular":
                sampled_items = set(
                    np.random.choice(
                        a=all_items,
                        size=num_to_sample,
                        replace=False,
                        p=all_item_popularities,
                    )
                )
            else:
                sampled_items = set(
                    np.random.choice(
                        a=all_items, size=num_to_sample, replace=False, p=None
                    )
                )

            # Add negative items with the ground truth item.
            sampled_items.add(session_ground_truth)

            # Create a sampled recommendation slate.
            # This conceptually corresponds to the model's ranking of the sampled items.
            sampled_session_predictions = [
                item for item in session_predictions if item in sampled_items
            ]
            sampled_session_predictions = sampled_session_predictions[:top_k]

            if len(sampled_session_predictions) < top_k:
                less_than_k_count += 1

            sampled_predictions[session_id] = sampled_session_predictions

        if less_than_l_count > 0:
            logger.warning(
                f"""[EVALUATION] For {less_than_l_count}/{len(ground_truths)}
                ground-truth samples there were less than {top_l} predictions."""
            )
        if less_than_k_count > 0:
            logger.warning(
                f"""[EVALUATION] For {less_than_k_count}/{len(ground_truths)}
                ground-truth samples there were less than {top_k} predictions."""
            )

        # Do conventional evaluation.
        report = Evaluation.eval(
            predictions=sampled_predictions,
            ground_truths=ground_truths,
            top_k=top_k,
            metrics=metrics.ALL_RANKING,
            dependencies=dependencies,
            cores=cores,
            model_name=model_name,
        )

        # Replace metric names.
        report.results = {
            f"sampled_{metric_name}": score
            for metric_name, score in report.results.items()
        }

        return report
