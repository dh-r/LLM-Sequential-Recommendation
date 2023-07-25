import logging
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any, Tuple
import numpy as np
from multiprocess import Pool
from main.data.abstract_dataset import Dataset, DatasetT
from main.eval import evaluation
from typing import Dict

import inspect
from enum import Enum

from main.abstract_model import Model


class MetricDependency(Enum):
    NUM_ITEMS = 0
    ITEM_COUNT = 1
    SAMPLE_COUNT = 2


class RankingMetric(ABC):
    """Abstract implementation of ranking metric in recommender systems.

    The metric can be evaluated in two ways:
    1. Directly evaluate against the predictions and ground truths:
        `RankingMetric.eval(predictions, ground_truths, top_k=5)`
    2. Run and evaluate a model on a dataset:
        `RankingMetric.run(model, dataset, top_k=5, use_folds=True)
    This allows the metric to directly apply cross-validation/testing, if necessary.

    Both implementations support multiprocessing by dividing the
    predictions/ground-truth combinations across the specified cores.
    """

    def __init__(self):
        """Initializes a new ranking metric.

        Note: do not initialize ranking metrics directly. Instead use .run() or .eval()
        on the classtype.
        """
        self.top_k: Optional[int] = None
        self.num_samples: Optional[int] = None
        self.num_items: Optional[int] = None
        self.dependencies: Dict[MetricDependency, Any] = {}

    @abstractmethod
    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> Union[float, Any]:
        """Evaluates a single sample of the metric.
        A sample could be a session, a user, a vendor, etc.

        Args:
            predictions (np.ndarray): The predictions/recommendations for this sample.
            ground_truth (np.ndarray): The ground truth/test labels for this sample.
            intersect (np.ndarray): The intersection between predictions and the
                ground-truth. This has been pre-computed as many metrics re-use this.
            sample_id (int): The sample id corresponding to the predictions,
                ground-truth and intersect.

        Returns:
            Union[float, Any]: The metric result for this single sample.
                Often a `float`, but it could be anything.
        """

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric."""
        pass

    def get_required_dependencies(self) -> List[MetricDependency]:
        """Override by a metric to specify its dependencies."""
        return []

    def eval_bulk(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        intersect: List[np.ndarray],
        sample_ids: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a bulk of samples.

        For some metrics bulk samples can be evaluated more efficiently using, for
        example, numpy matrix operations. In that case, this method can be overridden.
        Otherwise, it calls .eval_sample() for each individual sample.

        Note: all arguments preserve the same order.

        Args:
            predictions (List[np.ndarray]): The list of predictions.
            ground_truth (List[np.ndarray]): The list of ground truths.
            intersect (List[np.ndarray]): The list of the intersection between
                predictions and ground-truths.
            sample_ids (np.ndarray): A list of sample ids corresponding to the
                predictions, ground-truths and intersect.

        Returns:
            np.ndarray: A numpy array of the results of eval_sample.
        """
        # If there is no bulk implementation available.
        return np.array(
            [
                self.eval_sample(p, g, i, l)
                for p, g, i, l in zip(predictions, ground_truth, intersect, sample_ids)
            ]
        )

    @classmethod
    def eval(
        cls,
        predictions: Dict[int, np.ndarray],
        ground_truths: Dict[int, np.ndarray],
        top_k: int,
        dependencies: Dict[MetricDependency, Any] = {},
        cores: int = 1,
    ) -> Union[float, Any]:
        """Evaluates a single ranking metric.

        Args:
            predictions (Dict[int, np.ndarray]): The dictionary of predictions
                per sample. Complies with the output of each model.
            ground_truths (Dict[int, np.ndarray]): The dictionary of ground truths
                per sample.
            top_k (int): The cut-off point for the recommendations. It is not required
                that `predictions` has <= top_k predictions per sample. This method will
                only evaluate against the top_k predictions.
            dependencies (Dict[MetricDependency, Any], optional): Specifies the
                dependencies for this evaluation run. Some metrics require external
                dependencies, such as `item counts`, to compute the result. This dict
                can be used to specify certain dependencies. If not given, but
                required by a metric an exception will be thrown. In some scenarios,
                we are able to derive and ingest dependencies automatically.
            cores (int, optional): Compute this metric across multiple cores.
                Defaults to 1.

        Returns:
            Union[float, Any]: The result for this metric. Most of the time,
                averaged across all (test) samples.
        """
        # Prepare evaluation, ensures predictions and ground_truth are of equal length,
        # and that there are only top-k predictions.
        (
            predictions,
            ground_truths,
            intersect,
            sample_ids,
        ) = evaluation.Evaluation.prepare_evaluation(predictions, ground_truths, top_k)

        # Setup metric and its characteristics.
        metric = cls()
        num_samples = len(predictions)

        # Manage dependencies for a metric.
        # Each metric gets 'num_items', 'num_samples', 'top_k' by default.
        if MetricDependency.NUM_ITEMS not in dependencies:
            # We assume the union of test and predict data covers the item space.
            # We need to know this for metric such as catalog coverage.
            # It is good to keep in mind this is an approximation of the total items.
            logging.warning(
                f"NUM_ITEMS was not explicitly set, now deriving it from predictions and"
                f"ground truths."
            )
            num_items = evaluation.Evaluation.count_unique_items(
                predictions + ground_truths
            )
            metric.set_num_items(num_items)
        else:
            metric.set_num_items(dependencies[MetricDependency.NUM_ITEMS])

        metric.set_num_samples(num_samples)
        metric.set_top_k(top_k)

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

        # If we only compute on a single core, we simply compute the metric 'partially'
        # on the complete data and finalize the metric state.
        if cores == 1:
            metric_accumulated = metric.eval_partial(
                predictions, ground_truths, intersect, sample_ids
            )
            metric_final = metric.state_finalize(metric_accumulated)

            return metric_final

        # Prepare multi-processing.
        pool = Pool(processes=cores)
        q, r = divmod(len(predictions), cores)
        start = 0

        metrics_accumulated: List[Any] = []

        # Asynchronously evaluate work across cores.
        for i in range(cores):
            size = q + (i < r)
            end = start + size

            # Construct new object of this metric, and evaluate partially.
            metrics_accumulated.append(
                pool.apply_async(
                    metric.copy().eval_partial,
                    [
                        predictions[start:end],
                        ground_truths[start:end],
                        intersect[start:end],
                        sample_ids[start:end],
                        False,
                    ],
                )
            )
            start = end

        # Get per core, the intermediate metric result(s) and merge it across cores.
        metric_state = metric.state_init()
        for result_per_core in metrics_accumulated:
            metric_state = metric.state_merge(metric_state, result_per_core.get())

        # Get final result for the metric.
        metric_final = metric.state_finalize(metric_state)

        # Close multiprocessing pool.
        pool.close()

        return metric_final

    def eval_partial(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        intersect: List[np.ndarray],
        sample_ids: np.ndarray,
        metrics_per_sample: bool = False,
    ) -> Tuple[Any, Optional[np.ndarray]]:
        """Evaluates a metric partially given (a subset of) predictions
        and ground truths. This method is used to divide work across cores and does not
        average the metric results.

        Args:
            predictions (Dict[int, np.ndarray]): The dictionary of predictions
                per sample.
            ground_truths (Dict[int, np.ndarray]): The dictionary of ground truths
                per sample.
            intersect (List[np.ndarray]): The intersection between predictions
                and ground-truths.
            sample_ids (np.ndarray): A list of sample ids corresponding to the
                predictions, ground-truths and intersect.
            metrics_per_sample (bool, optional): If enabled, returns the metric result
                per sample. Defaults to False.

        Returns:
            Tuple[Any, Optional[List[Any]]]: The (accumulated) metric result and,
                if enabled, metric results per sample.
        """

        initial_state = self.state_init()
        bulk_result = self.eval_bulk(predictions, ground_truths, intersect, sample_ids)
        merged_state = self.state_merge_bulk(initial_state, bulk_result)

        # Since we are doing a 'partial' evaluation, we do not finalize
        # (e.g. average the metric state), instead we pass the intermediate state.
        if metrics_per_sample:
            if self.per_sample():
                return merged_state, bulk_result
            else:
                return merged_state, []

        return merged_state

    @classmethod
    def run(
        cls,
        model: Model,
        dataset: Dataset,
        top_k: int = 10,
        use_folds: bool = False,
        dependencies: Dict[MetricDependency, Any] = {},
        cores: int = 1,
    ) -> Union[float, Any]:
        """Runs a model and evaluate the metric.

        Returns:
            Union[float, Any]: The result for this metric. Most of the time, averaged
                across all (test) samples.
        """
        metric = cls()
        metric.set_top_k(top_k)

        if use_folds:
            total_folds = len(dataset.get_k_fold_eval())

            metric.set_num_samples(total_folds)  # trick to average results across folds
            metric.set_num_items(total_folds)

            metric_state = metric.state_init()
            for train_data, test_data_labels, test_data in dataset.get_k_fold_eval():
                metric_result = metric._run_model_and_eval(
                    model,
                    dataset=dataset,
                    train_data=train_data,
                    test_data_labels=test_data_labels,
                    test_data=test_data,
                    top_k=top_k,
                    dependencies=dependencies,
                    cores=cores,
                )
                metric_state = metric.state_merge(metric_state, metric_result)

            metric_result = metric.state_finalize(metric_state)
        else:
            train_data = dataset.train_data
            test_data_labels, test_data = dataset.get_test_data_eval()

            metric.set_num_samples(1)
            metric.set_num_items(1)

            metric_state = metric.state_init()
            metric_result = metric._run_model_and_eval(
                model,
                dataset=dataset,
                train_data=train_data,
                test_data_labels=test_data_labels,
                test_data=test_data,
                top_k=top_k,
                dependencies=dependencies,
                cores=cores,
            )
            metric_state = metric.state_merge(metric_state, metric_result)
            metric_result = metric.state_finalize(metric_state)

        return metric_result

    def _run_model_and_eval(
        self,
        model: Model,
        dataset: Dataset,
        train_data: DatasetT,
        test_data_labels: Any,
        test_data: Dict[int, np.ndarray],
        top_k: int,
        dependencies: Dict[MetricDependency, Any],
        cores: int,
    ) -> Union[float, Any]:
        """Utility method to run the actual model and evaluate its results.
        """
        model.train(train_data)

        recommendations = model.predict(test_data_labels, top_k=top_k)

        metric_result = self.eval(
            predictions=recommendations,
            ground_truths=test_data,
            top_k=top_k,
            cores=cores,
            dependencies={
                MetricDependency.NUM_ITEMS: dataset.get_unique_item_count(),
                MetricDependency.ITEM_COUNT: dataset.get_item_counts(),
                MetricDependency.SAMPLE_COUNT: dataset.get_sample_counts(),
            }
            | dependencies,
        )
        return metric_result

    def state_init(self) -> float:
        """Initializes the state of this metric to accumulate intermediate results.

        The default init state is 0.

        Returns:
            Any: The initial state of the metric.
        """
        return 0

    def state_merge(self, current: Any, to_add: Any) -> Any:
        """Merges intermediate metric results.
        This show how results per sample can be merged.

        By default, this sums the current state and the metric results.

        Args:
            current: The current state.
            to_add: The state to merge with.

        Returns:
            Any: The merged state.
        """
        return current + to_add

    def state_merge_bulk(self, current: Any, to_add: Any) -> Any:
        """Merges intermediate bulk metric results.
        This show how results of a set of samples can be merged.

        By default, this sums the current state and the sum of the metric results
        as an `np.nddaray`.

        Args:
            current: The current state.
            to_add: The bulk state to merge with.

        Returns:
            float: The merged state.
        """
        return current + to_add.sum()

    def state_finalize(self, current: float) -> float:
        """Finalizes the state.

        By default, this averages the sum of the results over the amount of samples.

        Args:
            current: The total state.

        Returns:
            Any: The finalized state.
        """
        if self.num_samples == 0:
            return 0

        return current / self.num_samples

    @staticmethod
    def _keyword_in_args(keyword: str, method) -> bool:
        """Check if a method has a keyword in its parameters.

        Args:
            keyword (str): The keyword to look for.
            method: The method. For example: instance.method_name.

        Returns:
            bool: True when method has keyword as parameter, otherwise False.
        """
        return keyword in inspect.getfullargspec(method)[0]

    def per_sample(self) -> bool:
        """Whether or not this metric is interpretable on a per-sample basis.

        At the evaluation level, one can define to return the metric results per user.
        However, for some metrics such as catalog coverage this is not relevant.
        In that case, return False in this method.
        """
        return True

    def copy(self) -> "RankingMetric":
        """Copies the ranking metric and is attributes.
        Used to pass a copy to all different cores, when multiprocessing is enabled.

        Returns:
            RankingMetric: A copied instance.
        """
        new_instance = self.__class__()
        new_instance.top_k = self.top_k
        new_instance.num_samples = self.num_samples
        new_instance.num_items = self.num_items
        new_instance.dependencies = self.dependencies

        return new_instance

    def set_top_k(self, top_k: int):
        """Set the top_k predictions to consider for this metric.

        Args:
            top_k (int): the top_k value.
        """
        self.top_k = top_k

    def set_num_samples(self, num_samples: int):
        """Set the number of samples used to evaluate this metric.

        Args:
            num_samples (int): The number of samples.
        """
        self.num_samples = num_samples

    def set_num_items(self, num_items: int):
        """The (unique) amount of items used to evaluate this metric.

        Args:
            num_items (int): The number of items.
        """
        self.num_items = num_items

    def set_dependencies(self, dependencies: Dict[MetricDependency, Any]):
        self.dependencies = dependencies

    def get_dependency(self, dep: MetricDependency) -> Any:
        if dep not in self.get_required_dependencies():
            raise AttributeError(
                f"{dep} is not in the required dependencies of this metric."
                f"Add it to the 'get_required_dependencies()' implementation."
            )
        if dep not in self.dependencies:
            raise AttributeError(f"{dep} has not been set.")

        return self.dependencies[dep]
