"""**************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 *************************************************************************"""

import sys, numpy as np
from  matplotlib import pyplot as plt

from Cabana_BenchmarkPlotUtils import *

# Plot all results in a list of files.
def plotAll(ax, data):
    color_dict = getColors(data)
    for backend in data.getAllBackends():
        for cat in data.getAllCategories():
            for type in data.getAllTypes():
                for param in data.getAllParams():
                    desc = ManualDataDescription(backend, type, cat, param)
                    result = AllSizesSingleResult(data, desc)

                    sizes = scaleSizes(result.sizes, data.grid)
                    plotResults(ax, sizes, result.times, backend, color_dict[cat])
    return False

# Compare host and device results from a list of files.
def plotCompareHostDevice(ax, data, compare="host"):
    color_dict = getColors(data)
    for backend in data.getAllBackends():
        if backend == compare: continue
        for cat in data.getAllCategories():
            for type in data.getAllTypes():
                for param in data.getAllParams():
                    desc = ManualDataDescription(backend, type, cat, param)
                    result = AllSizesSingleResult(data, desc)
                    desc2 = ManualDataDescription(compare, type, cat, param)
                    result2 = AllSizesSingleResult(data, desc2)

                    num_1 = len(result.times)
                    num_2 = len(result2.times)
                    max = num_1 if num_1 < num_2 else num_2

                    sizes = scaleSizes(result.sizes, data.grid)
                    speedup = result2.times / result.times
                    plotResults(ax, sizes, speedup, backend, color_dict[cat])
    return True

# Compare all results from two files. Optionally ignore some backends.
def plotCompareFiles(ax, data1, data2, ignore_backend=[]):
    color_dict = getColors(data1)
    backends1 = data1.getAllBackends()
    backends2 = data2.getAllBackends()
    for b in ignore_backend:
        if b in backends1: backends1.remove(b)
        if b in backends2: backends2.remove(b)
    for b1, b2 in zip(backends1, backends2):
        for c1, c2 in zip(data1.getAllCategories(), data2.getAllCategories()):
            for t1, t2 in zip(data1.getAllTypes(), data2.getAllTypes()):
                for p1, p2 in zip(data1.getAllParams(), data2.getAllParams()):
                    desc1 = ManualDataDescription(b1, t1, c1, p1)
                    result1 = AllSizesSingleResult(data1, desc1)
                    desc2 = ManualDataDescription(b2, t2, c2, p2)
                    result2 = AllSizesSingleResult(data2, desc2)

                    num_1 = len(result1.times)
                    num_2 = len(result2.times)
                    max = num_1 if num_1 < num_2 else num_2

                    sizes = scaleSizes(result1.sizes[:max], data1.grid)
                    speedup = result2.times[:max] / result1.times[:max]
                    plotResults(ax, sizes, speedup, b1, color_dict[c1])
    return True



if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit("Provide Cabana benchmark file path(s) on the command line.")
    filelist = sys.argv[1:]
    print(filelist)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ### Different options - plot all results or compare
    data = getData(filelist)

    speedup = plotAll(ax1, data)
    #speedup = plotCompareHostDevice(ax1, data, "serial")

    #data, data_f2 = getSeparateData(filelist)
    #speedup = plotCompareFiles(ax1, data, data_f2, ["cuda_host", "cudauvm_cudauvm", "hip_host"])
    ###

    createPlot(fig1, ax1, data,
               speedup=speedup, backend_label=True)#, cpu_name="POWER9", gpu_name="V100")
