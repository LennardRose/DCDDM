from numpy import ndarray
from river.base import DriftDetector, Classifier


class DCDDM(DriftDetector):
    """Domain Classifier Drift Detection Method for concept drift detection.

    Parameters
    ----------
    classifier
        Classifier to use as domain classifier.
    metric
        Metric to monitor the dissimilarity of both windows.
        Needs to be able to process the predictions of the domain classifier.
        Therefore, can be any metric that is able to process boolean values.
        If None is chosen, the dissimilarity is determined by the domain classifiers calculated probabilities.
    window_size
        Size of each the windows. Not the added size.
    threshold
        Threshold that determines how far the metric value can fluctuate around the reference value.
    fixed_threshold
        Determines if the threshold is fixed or not.
        For fixed threshold the borders are:
        $$ reference value +/- threshold$$
        For not fixed threshold the borders are calculated by:
        $$ reference value * (1 +/- threshold) $$
        Note that with this option, the threshold must be between 0 and 1.

    Notes
    -----
    DCDDM (Domain Classifier Drift Detection Method) is a concept drift detection method based
    on the predictive performance of a domain classifier. DCDDM can monitor data or performance distributions.
    It accepts single values as well as one dimensional input as dict.

    DCDDM maintains two windows $W_0$ (first_window) and $W_1$ (second_window) of fixed size $n$ (window_size). $W_1$
    either contains no data (at the beginning of detection) or the content of the last $W_1$. If both windows contain
    $n$ samples, the domain classifier learns each windows content with the samples of $W_0$ labeled as True and that
    of $W_1$ labeled as False, representing their respective domain. The domain classifier then tries to predict
    every samples domain. The performance (current_window_metric_value) of this test compared to previous performance
    (reference_value) indicates wether a drift is present or not. The performance test relies on the chosen metric.
    All metrics that are able to process boolean values are appropiate.


    Examples
    --------
    >>> import numpy as np
    >>> from river.linear_model import LogisticRegression
    >>> np.random.seed(12345)

    >>> dcddm = DCDDM(LogisticRegression())

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = np.concatenate((np.random.randint(2, size=1000),
    ...                               np.random.randint(4, high=8, size=1000)))

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     in_drift, in_warning = dcddm.update(val)
    ...     if in_drift:
    ...         print(f"Change detected at index {i - dcddm.window_size}, input value: {val}")
    ...         dcddm.reset()
    Change detected at index 999, input value: 6
    """


    def __init__(self, classifier: Classifier, metric=None, window_size=100, threshold=0.5, fixed_threshold=False):

        super().__init__()

        self.classifier = classifier

        self.metric = metric

        if fixed_threshold:
            self.threshold = threshold

        else:
            if 0 <= threshold <= 1:
                self.threshold = threshold
            else:
                raise ValueError("Relative threshold must be in [0;1]")

        self.fixed_threshold = fixed_threshold

        if window_size < 0:
            raise ValueError("window_size must be greater than 0")
        else:
            self.window_size = window_size
            self.two_windows_size = 2 * window_size

        self.first_window = []
        self.second_window = []

        self.start_or_drift_detected = True
        self.reference_value = None


    def update(self, value):
        """Update the  drift detector with a single data point.
        Adds an element to first_window. If only first_window is full, it gets shifted to second_window.
        If both windows are full, a test for concept drift is executed.

        Parameters
        ----------
        value
            New data to add to the first_window.

        Returns
        -------
        A tuple (drift, warning) where its elements indicate if a drift or a warning is detected.

        """
        current_window_metric_value = None

        # convert if necessary
        if type(value) != dict:
            value = convert_to_dict(value)

        # add value to window
        if len(self.first_window) < self.window_size:
            self.first_window.append(value)

        if len(self.first_window) == self.window_size:

            if self.second_window:
                self.learn_both_windows()

                if self.metric:
                    current_window_metric_value = self.compute_metric_value()
                else:
                    current_window_metric_value = self.compute_probability_ratio()

                self.shift_windows()

                if self.start_or_drift_detected:
                    self.reference_value = current_window_metric_value
                    self.start_or_drift_detected = False

            else:
                self.shift_windows()

        if len(self.first_window) > self.window_size:
            raise ValueError("Size of first window should not be bigger than window size")

        # detect change
        if current_window_metric_value:

            if self.is_outside_threshold(current_window_metric_value):
                self._in_concept_change = True
                self.start_or_drift_detected = True

            self.reset_classifier_and_metric()

        return self._in_concept_change, self._in_warning_zone


    def learn_both_windows(self):
        """
        classifier learns elements of both windows.
        first_window with domain-label True second_window with domain-label False
        """
        for value in self.first_window:
            self.classifier.learn_one(value, True)
        for value in self.second_window:
            self.classifier.learn_one(value, False)


    def compute_probability_ratio(self):
        """"
        Function used when no metric is specified. Returns ratio of the probabilities for each element of the
        two windows.
        """
        true = 0
        false = 0

        for window in [self.first_window, self.second_window]:
            for value in window:
                probability = self.classifier.predict_proba_one(value)
                true += probability[True]
                false += probability[False]

        if false == 0:
            false = 0.00001

        return true / false


    def compute_metric_value(self):
        """"
        Function used when metric is specified. Returns the score of the used metric.
        """
        # every element in the first window belongs to domain "True"
        for first_window_value in self.first_window:
            prediction = self.classifier.predict_one(first_window_value)
            self.metric = self.metric.update(True, prediction)

        # every element in the second window belongs to domain "False"
        for second_window_value in self.second_window:
            prediction = self.classifier.predict_one(second_window_value)
            self.metric = self.metric.update(False, prediction)

        return self.metric.get()


    def shift_windows(self):
        """
        Shift first_window content to the second_window
        first_window is empty afterwards
        """
        self.second_window = []
        self.second_window.extend(self.first_window)
        self.first_window = []


    def is_outside_threshold(self, value):
        """
        Checks if value is outside of threshold. Check depends on fixed_threshold
        :param value: the value to check
        :return: true if it is outside of threshold otherwise false
        """
        if self.fixed_threshold:
            return (value > self.reference_value + self.threshold
                    or value < self.reference_value - self.threshold)
        else:
            return (value > (self.reference_value * (1 + self.threshold))
                    or value < (self.reference_value * (1 - self.threshold)))


    def reset_classifier_and_metric(self):
        """
        Resets the classifier and the metric to be ready for the next window
        Does not reset DCDDM
        """
        self.classifier = self.classifier.clone()

        if self.metric:
            # self.metric = self.metric.clone() doesnt work dont know why
            metric_class = self.metric.__class__
            self.metric = metric_class()


    def reset(self):
        """
        Resets DCDDM
        Does not empty the second_window. If used on another dataset, second_window must be emptied manually.
        """
        super().reset()
        self.reset_classifier_and_metric()

        self.first_window = []
        self.reference_value = None
        self.start_or_drift_detected = True


# static
def convert_to_dict(value):
    """
    The classifiers learn(x) function needs input as dict, this function will convert list or single values to dicts
    :param value: the value that needs to be converted to a dict
    :return: the value as dict
    """
    if isinstance(value, ndarray) or hasattr(value, "__len__"):
        return {str(i): value[i] for i in range(0, len(value), 1)}
    else:
        return {"0": value}
