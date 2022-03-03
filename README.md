# DCDDM

DCDDM (Domain Classifier Drift Detection Method) is a concept drift
detection method based on the predictive performance of a domain
classifier. DCDDM can monitor data or performance distributions. DCDDM
maintains two windows *W*<sub>0</sub> and *W*<sub>1</sub> of fixed size
*n*. *W*<sub>1</sub> either contains no data (at the beginning of
detection) or the content of the last *W*<sub>1</sub>. If both windows
contain *n* samples, the domain classifier learns each windows content
with the samples of *W*<sub>0</sub> labeled as True and that of
*W*<sub>1</sub> labeled as False, representing their respective domain.
The domain classifier then tries to predict every samples domain. The
performance of this test compared to previous performance indicates
wether a drift is present or not.
