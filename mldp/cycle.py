# Copyright (c) 2018 Juniper Overbeck
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class Cycle(object):

    """Cycle class that contains testing instructions

    """

    def __init__(self, analyzer, handler, step, final=None, name="cycle"):
        """Instantiates a cycle

        :analyzer: Analyzer used to calculate statistics on measurements
        :handler: Data handler object
        :step: Function to be called upon every step
        :cleanup: Function to be called after all steps are finished

        """
        self._analyzer = analyzer
        self._handler = handler
        self._step = step
        self._final = final
        self._namme = name
        self._measurements = dict()

    def run(self):
        """Performs analysis on the internal data and generates analyses

        """
        for i, each in enumerate(self._handler):
            measurements = self._step(each)
            for k in measurements:
                self._measurements[k[0]+f'-{i}'] = k[1]
                self._analyzer.dump_measurement(k[0], k[1],
                                                f'{self._namme}-step-{i}')
        if self._final is not None:
            # This is gonna need a fixing, in order to properly deal with
            # the meta statistics its interested in.
            final_out = self._final(self._measurements)
            for i, j in final_out:
                # This behavior isn't correct
                # make a unit test to address this
                self._measurements[i] = j
        return self._measurements
