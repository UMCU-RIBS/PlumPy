'''
'''
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from plumpy.scripts.quality_checks import run_dqc
from plumpy.sigproc.general import ind2sec, sec2ind, resample
from plumpy.utils.io import load_config
from plumpy.utils.plots import *
import numpy as np
import pandas as pd
from numpy.random import RandomState
pd.set_option('display.max_rows', 500)


##
subj_cfg = '/Fridge/users/julia/project_corticom/cc2/14nav/cc2.yml'
subject = load_config(subj_cfg)
al_t, al_p = [], []
for run in subject['include_runs']:
    print(run)
    rest_data, rest_times, rest_events, _, rest_outliers = run_dqc(subj_cfg, 'rest', run)
    task_data, task_times, task_events, grid, task_outliers = run_dqc(subj_cfg, '14nav', run)

    ##
    task = '14nav'
    cfg = load_config(subject['tasks'][task])
    variant = cfg['order'][run]
    plot_path = subject['plot_path']
    data_path = subject['data_path']
    sr_post = cfg['target_sampling_rate']
    tag = f'{task}_{run}'


    ##
    plt.figure()
    plt.plot(task_data['hfb'][:, 5])
    plt.vlines(x=task_times, ymin=1, ymax=4, color='black')


    ############################ rest vs words hfb ###################################
    if variant == '1-7':
        rest_prep = [39]
        word_prep = list(range(32, 39))
        all_dict = {32: 'boven', 33: 'beneden', 34: 'omhoog', 35: 'omlaag', 36: 'links', 37: 'rechts', 38: 'selecteer', 39: 'rust'}
    elif variant == '8-14':
        rest_prep = [46]
        word_prep = list(range(39, 46))
        all_dict = {39: 'kiezen', 40: 'terug', 41: 'verwijder', 42: 'noord', 43: 'oost', 44: 'zuid', 45: 'west', 46: 'rust'}
    else:
        raise NotImplementedError
    all_prep = word_prep + rest_prep

    ## smooth data
    from plumpy.sigproc.general import smooth_signal_1d
    x = smooth_signal_1d(task_data['hfb'], n=15)
    y = smooth_signal_1d(rest_data['hfb'], n=15)

    ## z-score
    from plumpy.ml.general import zscore
    xz, scaler = zscore(x, y=y, xmin=rest_times[rest_events==0][0], xmax=rest_times[rest_events==1][0], units='samples')
    yz, _ = zscore(y, xmin=rest_times[rest_events==0][0], xmax=rest_times[rest_events==1][0], units='samples')

    ## save for classification
    # word onsets
    t_events_sec = np.array([ind2sec(i, sr_post) for i in task_times])
    onsets = pd.DataFrame({'text': np.array([all_dict[i] for i in task_events if i in all_dict.keys()]),
                  'xmin': np.array([t_events_sec[i] for i, j in enumerate(task_events) if j in all_dict.keys()])})

    onsets.to_csv(str(Path(data_path)/f'{tag}_all_prep_onsets.csv'), index=False)

    ## save processed excluding bad channels
    bad = [120]
    chan_indices = pd.DataFrame({'indices':np.setdiff1d(np.arange(x.shape[-1]), np.array(bad))})
    chan_indices.to_csv(str(Path(data_path) / f'{tag}_channel_indices.csv'), index=False)

    d_out = task_data.copy()
    for band in d_out.keys():
        t = np.delete(d_out[band], (bad), axis=1) # TODO: check for len(bad) > 1
        for sr in [100, 50, 10]:
            temp = resample(t, sr, sr_post)
            np.save(str(Path(data_path)/f'{tag}_car_{band}_{sr}Hz_nobad.npy'), temp)

            _, scaler2 = zscore(temp, xmin=t_events_sec[task_events == 50][0], duration=5, units='seconds', sr=sr)
            np.save(str(Path(data_path) / f'{tag}_car_{band}_{sr}Hz_nobad_precomputed_mean.npy'), scaler2.mean_)
            np.save(str(Path(data_path) / f'{tag}_car_{band}_{sr}Hz_nobad_precomputed_scale.npy'), scaler2.scale_)


    ## take fragments of rest from the separate rest task as a baseline to compare speech activity to
    prng2 = RandomState(599)
    ind_rest = np.sort(prng2.randint(rest_times[rest_events==0][0], rest_times[rest_events==1][0], 28))
    print(ind_rest)

    ## ttest words - separate rest
    for dur in [2 * sr_post, 4 * sr_post, 6 * sr_post]:
        tw = []
        for w in word_prep:
            if w in task_events:
                for iw in np.where(task_events == w)[0]:
                    tw.append(np.mean(xz[task_times[iw + 1]:task_times[iw + 1] + dur], 0))
        tw = np.array(tw)
        tr = []
        for r in ind_rest:
            tr.append(np.mean(yz[r:r + dur], 0))
        tr = np.array(tr)

        from scipy.stats import ttest_rel
        a = [ttest_rel(tw[:, i], tr[:, i]) for i in range(xz.shape[-1])]
        ts = np.array([i.statistic for i in a])
        ps = np.array([i.pvalue for i in a])
        plot_on_grid(grid, ts, label=f'Ttest words-rest ({int(dur/sr_post)} s)', colormap='vlag', xmin=-10, xmax=10)
        save_plot(plot_path, name=tag + f'_ttest_words_sep_rest_dur{int(dur/sr_post)}s_pale')
        plot_on_grid(grid, ts-np.mean(ts), label=f'Ttest words-rest ({int(dur/sr_post)} s)', colormap='vlag', xmin=-10, xmax=10)
        save_plot(plot_path, name=tag + f'_ttest_words_sep_rest_dur{int(dur/sr_post)}s_pale_demeant')
        if dur == 6 * sr_post:
            al_t.append(ts)
            al_p.append(ps)

    ## plot averages over all words: rest rest
    for av_words in [True, False]:
        with sns.plotting_context('poster', font_scale=1):
            for gid, add in zip([3, 4, 2, 1], [0, 4, 8, 12]):
                fig = plt.figure(figsize=(16, 10), layout="constrained")
                spec = fig.add_gridspec(nrows=4, ncols=8)
                for i in range(4):
                    for j in range(8):
                        ax = fig.add_subplot(spec[i, j])
                        if av_words:
                            tw = []
                            for w in word_prep:
                                for iw in np.where(task_events == w)[0]:
                                    tw.append(xz[task_times[iw + 1]:task_times[iw + 1] + 4 * sr_post])
                            [plt.plot(it[:, grid[i+add, j] - 1], color='#1f77b4', linewidth=.2, alpha=.15) for it in tw]
                        else:
                            for w in word_prep:
                                tw = []
                                if w in task_events:
                                    for iw in np.where(task_events == w)[0]:
                                        tw.append(xz[task_times[iw + 1]:task_times[iw + 1] + 4 * sr_post])
                                    [plt.plot(it[:, grid[i + add, j] - 1], color='#1f77b4', linewidth=.2, alpha=.15) for it in tw]
                                    plt.plot(np.mean(np.array(tw), 0)[:, grid[i + add, j] - 1], color='#1f77b4', linewidth=1)
                        tr = []
                        for r in ind_rest:
                            tr.append(yz[r:r + 4 * sr_post])
                        [plt.plot(it[:, grid[i+add, j] - 1], color='red', linewidth=.2, alpha=.15) for it in tr]
                        plt.plot(np.mean(np.array(tr), 0)[:, grid[i+add, j] - 1], color='red', linewidth=1)
                        if av_words:
                            plt.plot(np.mean(np.array(tw), 0)[:, grid[i + add, j] - 1], color='#1f77b4', linewidth=1)
                        plt.title(grid[i+add, j])
                        plt.ylim(-3.5, 3.5)
                        ax.set(xticklabels=[])  # remove the tick labels
                        ax.tick_params(bottom=False)
                        if not (i == 0 and j == 0):
                            ax.set(yticklabels=[])  # remove the tick labels
                            ax.tick_params(left=False)
                if av_words:
                    save_plot(plot_path, name=tag + f'_sep_rest_hfb_word_rest_avg_words_grid{gid}')
                else:
                    save_plot(plot_path, name=tag + f'_sep_rest_hfb_word_rest_sep_words_grid{gid}')

    ## plot averages over all words: task rest
    for av_words in [True, False]:
        with sns.plotting_context('poster', font_scale=1):
            for gid, add in zip([3, 4, 2, 1], [0, 4, 8, 12]):
                fig = plt.figure(figsize=(16, 10), layout="constrained")
                spec = fig.add_gridspec(nrows=4, ncols=8)
                for i in range(4):
                    for j in range(8):
                        ax = fig.add_subplot(spec[i, j])
                        if av_words:
                            tw = []
                            for w in word_prep:
                                for iw in np.where(task_events == w)[0]:
                                    tw.append(xz[task_times[iw + 1]:task_times[iw + 1] + 4 * sr_post])
                            [plt.plot(it[:, grid[i+add, j] - 1], color='#1f77b4', linewidth=.2, alpha=.15) for it in tw]
                        else:
                            for w in word_prep:
                                tw = []
                                if w in task_events:
                                    for iw in np.where(task_events == w)[0]:
                                        tw.append(xz[task_times[iw + 1]:task_times[iw + 1] + 4 * sr_post])
                                    [plt.plot(it[:, grid[i + add, j] - 1], color='#1f77b4', linewidth=.2, alpha=.15) for it in tw]
                                    plt.plot(np.mean(np.array(tw), 0)[:, grid[i + add, j] - 1], color='#1f77b4', linewidth=1)
                        tr = []
                        for r in rest_prep:
                            for ir in np.where(task_events == r)[0]:
                                tr.append(xz[task_times[ir + 1]:task_times[ir + 1] + 4 * sr_post])
                        [plt.plot(it[:, grid[i+add, j] - 1], color='red', linewidth=.2, alpha=.15) for it in tr]
                        plt.plot(np.mean(np.array(tr), 0)[:, grid[i+add, j] - 1], color='red', linewidth=1)
                        if av_words:
                            plt.plot(np.mean(np.array(tw), 0)[:, grid[i + add, j] - 1], color='#1f77b4', linewidth=1)
                        plt.title(grid[i+add, j])
                        plt.ylim(-3.5, 3.5)
                        ax.set(xticklabels=[])  # remove the tick labels
                        ax.tick_params(bottom=False)
                        if not (i == 0 and j == 0):
                            ax.set(yticklabels=[])  # remove the tick labels
                            ax.tick_params(left=False)
                if av_words:
                    save_plot(plot_path, name=tag + f'_task_rest_hfb_word_rest_avg_words_grid{gid}')
                else:
                    save_plot(plot_path, name=tag + f'_task_rest_hfb_word_rest_sep_words_grid{gid}')

    ## plot averages separate per word: task rest
    # gid = 1
    # add = 12
    for gid, add in zip([3, 4, 2, 1], [0, 4, 8, 12]):
        fig = plt.figure(figsize=(16, 10), layout="constrained")
        spec = fig.add_gridspec(nrows=4, ncols=8)
        for i in range(4):
            for j in range(8):
                ax = fig.add_subplot(spec[i, j])
                for w in word_prep:
                    tw = []
                    if w in task_events:
                        for iw in np.where(task_events == w)[0]:
                            tw.append(xz[task_times[iw + 1]:task_times[iw + 1] + 4 * sr_post])
                        [plt.plot(it[:, grid[i+add, j] - 1], color='#1f77b4', linewidth=.2, alpha=.15) for it in tw]
                        plt.plot(np.mean(np.array(tw), 0)[:, grid[i + add, j] - 1], color='#1f77b4', linewidth=1)
                tr = []
                for r in rest_prep:
                    for ir in np.where(task_events == r)[0]:
                        tr.append(xz[task_times[ir + 1]:task_times[ir + 1] + 4 * sr_post])
                [plt.plot(it[:, grid[i+add, j] - 1], color='red', linewidth=.2, alpha=.15) for it in tr]
                plt.plot(np.mean(np.array(tr), 0)[:, grid[i+add, j] - 1], color='red', linewidth=1)
                plt.title(grid[i+add, j])
                plt.ylim(-3.5, 3.5)
                ax.set(xticklabels=[])  # remove the tick labels
                ax.tick_params(bottom=False)
                if not (i == 0 and j == 0):
                    ax.set(yticklabels=[])  # remove the tick labels
                    ax.tick_params(left=False)
        save_plot(plot_path, name=tag + f'_hfb_word_rest_sep_words_grid{gid}')

    plt.close('all')


    plot_on_grid(grid, np.median(np.array(al_t),0), label=f'Ttest words-rest ({int(dur / sr_post)} s)', colormap='vlag', xmin=-10, xmax=10)
    save_plot(plot_path, name=f'_median_ttest_words_sep_rest_dur{int(dur / sr_post)}s_pale')

    ###################################################################################################################
