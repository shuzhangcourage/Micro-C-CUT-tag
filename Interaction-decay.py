# import core packages
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations

# import semi-core packages
import matplotlib.pyplot as plt
from matplotlib import colors
plt.style.use('seaborn-poster')
import numpy as np
import pandas as pd

# import open2c libraries
import bioframe

import cooler
import cooltools

from packaging import version
if version.parse(cooltools.__version__) < version.parse('0.5.0'):
    raise AssertionError("tutorials rely on cooltools version 0.5.0 or higher,"+
                         "please check your cooltools version and update to the latest")

# Load a Hi-C map at a 1kb resolution from a cooler file.
resolution = 1000 # note this might be slightly slow on a laptop
                  # and could be lowered to 10kb for increased speed
clr = cooler.Cooler('./DLD1_async_14hcontrol_matrix_1kb.mcool::/resolutions/'+str(resolution))
aux = cooler.Cooler('./DLD1_async_14hdegron_matrix_1kb.mcool::/resolutions/'+str(resolution))
HiC = cooler.Cooler('./mergeG1_control_DLD1_RNAPll_mAID_NadineShu.1kb.mcool::/resolutions/'+str(resolution))
# Use bioframe to fetch the genomic features from the UCSC.
hg38_chromsizes = bioframe.fetch_chromsizes('hg38')
hg38_arms = bioframe.core.construction.add_ucsc_name_column(bioframe.make_viewframe(hg38_chromsizes))
hg38_arms = hg38_arms[hg38_arms.chrom.isin(clr.chromnames)].reset_index(drop=True)
ctrl_cvd_smooth_agg = cooltools.expected_cis(
    clr=clr,
    view_df=hg38_arms,
    smooth=True,
    aggregate_smoothed=True,
    nproc=20
)
aux_cvd_smooth_agg = cooltools.expected_cis(
	clr=aux,
	view_df=hg38_arms,
	smooth=True,
	aggregate_smoothed=True,
	nproc=20
)
HiC_cvd_smooth_agg = cooltools.expected_cis(
    clr=HiC,
    view_df=hg38_arms,
    smooth=True,
    aggregate_smoothed=True,
    nproc=20
)
ctrl_cvd_smooth_agg['s_bp'] = ctrl_cvd_smooth_agg['dist']* resolution
ctrl_cvd_smooth_agg['balanced.avg.smoothed.agg'].loc[ctrl_cvd_smooth_agg['dist'] < 2] = np.nan
aux_cvd_smooth_agg['s_bp'] = aux_cvd_smooth_agg['dist']* resolution
aux_cvd_smooth_agg['balanced.avg.smoothed.agg'].loc[aux_cvd_smooth_agg['dist'] < 2] = np.nan
HiC_cvd_smooth_agg['s_bp'] = HiC_cvd_smooth_agg['dist']* resolution
HiC_cvd_smooth_agg['balanced.avg.smoothed.agg'].loc[HiC_cvd_smooth_agg['dist'] < 2] = np.nan
# Just take a single value for each genomic separation

ctrl_cvd_merged = ctrl_cvd_smooth_agg.drop_duplicates(subset=['dist'])[['s_bp', 'balanced.avg.smoothed.agg']]
aux_cvd_merged = aux_cvd_smooth_agg.drop_duplicates(subset=['dist'])[['s_bp', 'balanced.avg.smoothed.agg']]
HiC_cvd_merged = HiC_cvd_smooth_agg.drop_duplicates(subset=['dist'])[['s_bp', 'balanced.avg.smoothed.agg']]
# Calculate derivative in log-log space
ctrl_der = np.gradient(np.log(ctrl_cvd_merged['balanced.avg.smoothed.agg']),
                  np.log(ctrl_cvd_merged['s_bp']))
aux_der = np.gradient(np.log(aux_cvd_merged['balanced.avg.smoothed.agg']),
                  np.log(aux_cvd_merged['s_bp']))
HiC_der = np.gradient(np.log(HiC_cvd_merged['balanced.avg.smoothed.agg']),
                  np.log(HiC_cvd_merged['s_bp']))
f, axs = plt.subplots(
    figsize=(6.5,13),
    nrows=2,
    gridspec_kw={'height_ratios':[6,2]},
    sharex=True)
ax = axs[0]
ax.loglog(
    ctrl_cvd_merged['s_bp'],
    ctrl_cvd_merged['balanced.avg.smoothed.agg'],
    '-',
    markersize=5,
)

ax.set(
    ylabel='IC contact frequency',
    xlim=(1e3,1e8)
)
ax.set_aspect(1.0)
ax.grid(False)

ax.loglog(
    aux_cvd_merged['s_bp'],
    aux_cvd_merged['balanced.avg.smoothed.agg'],
    '-',
    markersize=5,
)

ax.loglog(
    HiC_cvd_merged['s_bp'],
    HiC_cvd_merged['balanced.avg.smoothed.agg'],
    '-',
    markersize=5,
)

ax.legend(['Ctrl_Micro-C', 'Aux_Micro-C','MergeG1_Hi-C'], loc='upper right')

ax = axs[1]
ax.semilogx(
    ctrl_cvd_merged['s_bp'],
    ctrl_der,
    alpha=0.5
)

ax.set(
    xlabel='separation, bp',
    ylabel='slope')

ax.grid(False)
ax.semilogx(
    aux_cvd_merged['s_bp'],
    aux_der,
    alpha=0.5
)

ax.semilogx(
    HiC_cvd_merged['s_bp'],
    HiC_der,
    alpha=0.5
)

plt.savefig("Interaction_Decay.pdf", dpi='figure', format=None, metadata=None)
