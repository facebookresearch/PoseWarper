from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from poseval.py.evaluateAP import evaluateAP
from poseval.py.evaluateTracking import evaluateTracking
from poseval.py.eval_helpers import Joint, printTable, load_data_dir, getCum


def evaluate(gtdir, preddir, eval_pose=True, eval_track=True,
             eval_upper_bound=False):
    gtFramesAll, prFramesAll = load_data_dir(['', gtdir, preddir])

    print('# gt frames  :', len(gtFramesAll))
    print('# pred frames:', len(prFramesAll))

    apAll = np.full((Joint().count + 1, 1), np.nan)
    preAll = np.full((Joint().count + 1, 1), np.nan)
    recAll = np.full((Joint().count + 1, 1), np.nan)
    if eval_pose:
        apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll)
        print('Average Precision (AP) metric:')
        #printTable(apAll)
        cum = printTable(apAll)

    metrics = np.full((Joint().count + 4, 1), np.nan)
    #print(eval_track)
    if eval_track:
        #print(xy)
        metricsAll = evaluateTracking(
            gtFramesAll, prFramesAll, eval_upper_bound)

        for i in range(Joint().count + 1):
            metrics[i, 0] = metricsAll['mota'][0, i]
        metrics[Joint().count + 1, 0] = metricsAll['motp'][0, Joint().count]
        metrics[Joint().count + 2, 0] = metricsAll['pre'][0, Joint().count]
        metrics[Joint().count + 3, 0] = metricsAll['rec'][0, Joint().count]
        print('Multiple Object Tracking (MOT) metrics:')
        track_cum = printTable(metrics, motHeader=True)
    #return (apAll, preAll, recAll), metrics
    #print(xy)
    return cum
