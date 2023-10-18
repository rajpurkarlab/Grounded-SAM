Download the CheXlocalize dataset [here](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c).

## Calculate human benchmark metrics on CheXlocalize test set

There are 668 chest X-rays from 500 patients in the CheXlocalize test set.

We remove the CXR associated with id `patient65086_study1_view1_frontal` because there are no ground-truth segmentations recorded even though there are positive labels for this CXR.

We remove the CXR associated with id `patient65145_study1_view1_frontal` because there are no human benchmark segmentations even though there are ground-truth segmentations. Since human benchmark radiologists were given the ground-truth labels, we suspect that it is an error that it's missing from hb_dict.

We remove the CXRs associated with ids `patient64860_study1_view1_frontal`, `patient64860_study1_view2_lateral`, and `patient65192_study1_view1_frontal` because there are no human benchmark segmentations for them even though their ids are included in `hb_dict`.

Therefore, we have:
- 666 CXRs in the test set
- 498 CXRs with ground-truth segmentations
- 498 CXRs with human benchmark segmentations

To calculate the metrics, use the following command:

```
(chexlocalize) > python human_benchmark_metrics.py \
    --gt_path gt_segmentations_test.json \
    --hb_path hb_segmentations_test.json \
    --gt_labels_path test_labels.csv
```

Both `gt_segmentations_test.json` and `hb_segmentations_test.json` are formatted as follows:
```
{
    'patient64622_study1_view1_frontal': {
	    'Enlarged Cardiomediastinum': {
		'size': [2320, 2828], # (h, w)
		'counts': '`Vej1Y2iU2c0B?F9G7I6J5K6J6J6J6J6H8G9G9J6L4L4L4L4L3M3M3M3L4L4...'},
	    ....
	    'Support Devices': {
		'size': [2320, 2828], # (h, w)
		'counts': 'Xid[1R1ZW29G8H9G9H9F:G9G9G7I7H8I7I6K4L5K4L5K4L4L5K4L5J5L5K...'}
    },
    ...
    'patient64652_study1_view1_frontal': {
	...
    }
}
```
Each file include only those CXRs with at least one positive ground-truth label. However, each CXR id key has values for all ten pathologies (regardless of ground-truth label).