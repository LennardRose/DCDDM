# DCDDM

DCDDM (Domain Classifier Drift Detection Method) is a concept drift detection method based
on the predictive performance of a domain classifier. DCDDM can monitor data or performance distributions.
DCDDM maintains two windows $W_0$ and $W_1$ of fixed size $n$. 
$W_1$ either contains no data (at the beginning of detection) or the content of the last $W_1$. If both windows contain
$n$ samples, the domain classifier learns each windows content with the samples of $W_0$ labeled as True and that
of $W_1$ labeled as False, representing their respective domain. The domain classifier then tries to predict
every samples domain. The performance of this test compared to previous performance indicates wether a drift is present or not. 
